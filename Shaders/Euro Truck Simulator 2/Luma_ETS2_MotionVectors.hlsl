// ETS2 DLAA: reconstruct motion vectors from #6127's packed depth buffer.
//
// 1:1 translation of the reprojection math in PS 0x805A8981 (TAA resolve).
// Inputs:
//   t0 = SRV4, R16G16_UINT written by #6127 (0xB824672E)
//          .x = log(minDepth + 1) * 4096.118164
//          .y = [bits 0..1] 3x3 min X offset - 1
//               [bits 2..3] 3x3 min Y offset - 1
//               [bits 4.. ] second log-quantized depth (not needed for MV)
//   b0 = a 256-byte copy of the game's CB0 (16 x float4). The engine
//        suballocates a 2 MB ring buffer with a dynamic pFirstConstant, so
//        C++ reads PSGetConstantBuffers1() and copies the active 256-byte
//        window into this self-contained cbuffer instead of binding the
//        game's buffer directly.
//
// Output is a UV-space MV (historyUV - currentUV). NGX converts it to pixel
// space via InMVScaleX/InMVScaleY, exposed as mvs_x_scale / mvs_y_scale.

cbuffer GameCB0 : register(b0)
{
   float4 cb0[16];
};

Texture2D<uint2> PackedDepth : register(t0);

float2 main(float4 pos : SV_Position) : SV_Target0
{
   const int2 pixel = int2(pos.xy);
   const uint2 packed = PackedDepth.Load(int3(pixel, 0));

   // 0x805A8981: utof / mul 0.000244 / exp (D3D ISA exp is 2^x = exp2!) / add -1
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

   // 0x805A8981: dp4 with the previous-frame view-projection rows 0/1/3
   const float clip_x = dot(cb0[0], ray);
   const float clip_y = dot(cb0[1], ray);
   const float clip_w = dot(cb0[3], ray);

   // 0x805A8981: max |xy| ; lt clip.w ; ge 0, clip.w ; or ; if_z
   const bool outside = clip_w <= max(abs(clip_x), abs(clip_y)) || clip_w <= 0.f;
   if (outside)
      return float2(0.f, 0.f); // movc falls back to the current UV => MV = 0

   // 0x805A8981: div by clip.w ; mad (0.5,-0.5) + cb0[8].xy ; add 0.5
   const float2 ndc = float2(clip_x, clip_y) / clip_w;
   const float2 reprojected_uv = ndc * float2(0.5f, -0.5f) + cb0[8].xy + 0.5f;

   // 0x805A8981: mad r1.xz, reproj, cb0[10].xy, -r1.xz ; add v1.xy
   // r1.xz was neighbour_uv, so the +uv and -uv cancel out.
   const float2 history_uv = reprojected_uv * cb0[10].xy - neighbour * cb0[9].yz;

   return history_uv - uv;
}
