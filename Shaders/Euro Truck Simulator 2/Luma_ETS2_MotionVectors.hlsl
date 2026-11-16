// ETS2 DLAA: reconstruct motion vectors from #6127's packed depth buffer
// and GBuffer flags (SRV 15) to separate the vehicle cabin from the world.
//
// Inputs:
//   t0 = SRV4, R16G16_UINT written by #6127 (0xB824672E)
//          .x = log2(minDepth + 1) * 4096.118164
//          .y = [bits 0..1] 3x3 min X offset - 1
//               [bits 2..3] 3x3 min Y offset - 1
//   t1 = SRV15, R16G16B16A16_UINT GBuffer flags written by geometry pass
//          .w [bit 13] dynamic / vehicle object flag (0x805A8981 disasm line 39 & 100)
//   b0 = a 256-byte copy of the game's CB0 (16 x float4)

cbuffer GameCB0 : register(b0)
{
   float4 cb0[16];
};

Texture2D<uint2> PackedDepth : register(t0);
Texture2D<uint4> GBufferFlags : register(t1);

float2 main(float4 pos : SV_Position) : SV_Target0
{
   const int2 pixel = int2(pos.xy);
   const uint2 packed = PackedDepth.Load(int3(pixel, 0));

   // 0x805A8981: utof / mul 0.000244 / exp2 / add -1
   const float depth = exp2(float(packed.x) * (1.f / 4096.118164f)) - 1.f;

   // 0x805A8981: and / ubfe / iadd -1 / itof
   const float2 neighbour = float2(float(packed.y & 3u) - 1.f,
                                   float((packed.y >> 2) & 3u) - 1.f);

   // 0x805A8981: div l(1), cb0[11].zw ; mul cb0[11].xy  => render size
   const float2 render_size = cb0[11].xy / max(cb0[11].zw, 1e-6f);
   const float2 uv = pos.xy / max(render_size, 1e-6f); // v1.xy

   // 0x805A8981: mad r1.xz, ..., cb0[9].yyzy, v1  (x uses .y, y uses .z)
   const float2 neighbour_uv = neighbour * cb0[9].yz + uv;

   // 0x805A8981: mul cb0[10].zw ; mad cb0[13].zw, cb0[13].xy
   float2 ray_xy = neighbour_uv * cb0[10].zw;
   ray_xy = ray_xy * cb0[13].zw + cb0[13].xy;

   // 0x805A8981: mov r4.zw, (1,1) ; mul r4.xyz, -depth
   float4 ray = float4(ray_xy, 1.f, 1.f);
   ray.xyz *= -depth;

   // Read GBuffer flags (SRV 15, R16G16B16A16_UINT):
   // 0x805A8981 lines 39 & 100-101: bit 13 of .w flags dynamic / vehicle geometry
   // 0x805A8981 lines 39-42 & 67 & 262: bit 0 of .z flags rearview mirror reflection surfaces
   const uint4 gbuffer = GBufferFlags.Load(int3(pixel, 0));
   const bool is_vehicle_flag = ((gbuffer.w >> 13) & 1u) != 0;
   const bool is_mirror = (gbuffer.z & 1u) != 0;

   // 1. Rearview mirror reflection:
   // Mirrors reflect the world behind the truck. Planar reflections have NO valid
   // screen-space motion vectors from the main camera's depth buffer.
   // If assigned cabin MV=(0,0), DLSS accumulates 16+ frames of moving reflection at 0 offset,
   // causing heavy blur and ghosting.
   // Returning an off-screen motion vector signals DLSS that history is disoccluded,
   // forcing sharp, fresh current-frame reconstruction with ZERO temporal blur!
   if (is_mirror)
   {
      return float2(2.0f, 2.0f);
   }

   // 2. Cabin / vehicle geometry:
   // Rigidly attached to the camera in first-person view.
   // Must ONLY use is_vehicle_flag within 8 meters.
   // NEVER use a raw (depth < 3.5f) threshold without is_vehicle_flag, because the
   // road directly in front of the truck (depth 1.5m~3.5m) would get MV=0,
   // causing violent tearing and shaking against the rest of the road!
   const bool is_cabin = is_vehicle_flag && (depth < 8.0f);

   float clip_x, clip_y, clip_w;
   if (is_cabin)
   {
      // Pure rotation reprojection: zero out translation column (.w = 0)
      clip_x = dot(float4(cb0[0].xyz, 0.f), ray);
      clip_y = dot(float4(cb0[1].xyz, 0.f), ray);
      clip_w = dot(float4(cb0[3].xyz, 0.f), ray);
   }
   else
   {
      // Full camera reprojection (translation + rotation) for the outside world
      clip_x = dot(cb0[0], ray);
      clip_y = dot(cb0[1], ray);
      clip_w = dot(cb0[3], ray);
   }

   // 0x805A8981: max |xy| ; lt clip.w ; ge 0, clip.w ; or ; if_z
   const bool outside = clip_w <= max(abs(clip_x), abs(clip_y)) || clip_w <= 0.f;
   if (outside)
      return float2(0.f, 0.f); // movc falls back to the current UV => MV = 0

   // 0x805A8981: div by clip.w ; mad (0.5,-0.5) ; add 0.5
   // Note: cb0[8].xy (jitter) is removed so MVs are pure and jitter-free for DLSS
   const float2 ndc = float2(clip_x, clip_y) / clip_w;
   const float2 reprojected_uv = ndc * float2(0.5f, -0.5f) + 0.5f;

   // 0x805A8981: mad r1.xz, reproj, cb0[10].xy, -r1.xz ; add v1.xy
   const float2 history_uv = reprojected_uv * cb0[10].xy - neighbour * cb0[9].yz;

   return history_uv - uv;
}
