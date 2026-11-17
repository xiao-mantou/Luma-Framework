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
//          .z [bit 0]  rearview mirror & dynamic electronic screen flag (0x805A8981 disasm line 170 & 296)
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
   // Disassembly of ETS2 TAA resolve 0x805A8981:
   //   bit 13 of .w flags vehicle / dynamic mesh geometry
   //   bit 0 of .z flags rearview mirror reflection surfaces and screens
   const uint4 gbuffer = GBufferFlags.Load(int3(pixel, 0));
   const bool is_vehicle_flag = ((gbuffer.w >> 13) & 1u) != 0;

   // Detect if camera and vehicle are completely static (zero translation and rotation in CB0)
   const bool is_static_frame = 
      (abs(cb0[0].w) < 1e-4f) && (abs(cb0[1].w) < 1e-4f) && (abs(cb0[3].w) < 1e-4f) &&
      (abs(cb0[0].y) < 1e-4f) && (abs(cb0[0].z) < 1e-4f) &&
      (abs(cb0[1].x) < 1e-4f) && (abs(cb0[1].z) < 1e-4f) &&
      (abs(cb0[3].x) < 1e-4f) && (abs(cb0[3].y) < 1e-4f);

   // Rearview mirror reflection surface (physical mirror glass & electronic screens):
   // Disassembly of ETS2 TAA resolve 0x805A8981 (lines 170 & 296):
   //   bit 0 of .z marks rearview mirror reflection surfaces and screens.
   const bool is_mirror_glass = ((gbuffer.z & 1u) != 0u) && (depth < 4.0f);
   if (is_mirror_glass && !is_static_frame)
   {
      // When moving, planar reflection moves at 80 km/h independent of the camera depth.
      // An off-screen disocclusion vector forces DLSS to discard stale history accumulation,
      // completely eliminating temporal smearing inside the mirror reflection!
      return float2(2.0f, 2.0f);
   }

   // Cabin geometry:
   // 1. GBuffer t15 bit 13 marks vehicle geometry within 8 meters.
   // 2. Comprehensive depth < 3.6m fallback guarantees that ALL cabin interior parts
   //    (steering wheel, dashboard, gauges, pillars, passenger seat, right mirror, sun visors)
   //    are 100% captured, even if accessory meshes lack bit 13 in GBuffer.
   //    (In ETS2 cabin view, distance to windshield is ~1.5m, passenger side is ~2.5m, right mirror is ~3.0m.
   //     The closest visible ground in front of the truck is >4.5m away, so depth < 3.6m never touches the road).
   const bool is_cabin = (is_vehicle_flag && depth < 8.0f) || (depth < 3.6f);

   float clip_x, clip_y, clip_w;
   if (is_cabin)
   {
      // Pure rotation with unit-depth ray (w = 0 strips translation).
      // ray.xyz was -depth * (ray_xy, 1.0f).
      // With depth = 1.0f and w = 0, unit_ray is (-ray_xy, -1.0f, 0.0f).
      // Stripping translation (.w = 0) eliminates world forward motion, while
      // preserving exact angular motion for head rotation and braking/sway pitch.
      const float4 unit_ray = float4(-ray_xy, -1.f, 0.f);
      clip_x = dot(cb0[0], unit_ray);
      clip_y = dot(cb0[1], unit_ray);
      clip_w = max(dot(cb0[3], unit_ray), 1e-4f);
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
      return is_static_frame ? float2(0.f, 0.f) : float2(2.0f, 2.0f);

   // 0x805A8981 lines 74-75:
   //   mad r5.xy, r5.xy, float2(0.5, -0.5), cb0[8].xy
   //   add r5.xy, r5.xy, float2(0.5, 0.5)
   // Crucial: cb0[8].xy is added to cancel the projection jitter between frames!
   // Without +cb0[8].xy, the reprojection matrix leaves ~0.58px of residual jitter in the MV,
   // causing high-frequency screen vibration in static scenes.
   const float2 ndc = float2(clip_x, clip_y) / clip_w;
   const float2 reprojected_uv = ndc * float2(0.5f, -0.5f) + 0.5f + cb0[8].xy;

   // 0x805A8981: mad r1.xz, reproj, cb0[10].xy, -r1.xz ; add v1.xy
   const float2 history_uv = reprojected_uv * cb0[10].xy - neighbour * cb0[9].yz;

   return history_uv - uv;
}
