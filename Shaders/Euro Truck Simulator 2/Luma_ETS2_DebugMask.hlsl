// ETS2 DLAA: Debug Visual Mask for inspecting GBuffer Flags and Depth
//
// Inputs:
//   t0 = SRV15, R16G16B16A16_UINT GBuffer flags written by geometry pass
//          .w [bit 13] dynamic / vehicle object flag (0x805A8981 disasm line 39 & 100)
//          .z [bit 0]  rearview mirror & dynamic electronic screen flag (0x805A8981 disasm line 170 & 296)
//   t1 = SRV4,  R16G16_UINT written by #6127 (0xB824672E) - Packed Depth
//          .x = log2(minDepth + 1) * 4096.118164
//   t2 = SRV1,  RGBA16F original jittered scene color (used if BLEND_OVER_GAME == 1)
//   b0 = GameCB0 (16 x float4)

// Set to 1 to blend colors over the live game scene (50% opacity), or 0 for solid high-contrast mask colors
#define BLEND_OVER_GAME 0

cbuffer GameCB0 : register(b0)
{
   float4 cb0[16];
};

Texture2D<uint4> GBufferFlags : register(t0);
Texture2D<uint2> PackedDepth : register(t1);
Texture2D<float4> SceneColor : register(t2);

float4 main(float4 pos : SV_Position) : SV_Target0
{
   const int2 pixel = int2(pos.xy);
   const uint4 gbuffer = GBufferFlags.Load(int3(pixel, 0));
   const uint2 packed = PackedDepth.Load(int3(pixel, 0));

   // 0x805A8981 depth decode:
   const float depth = exp2(float(packed.x) * (1.f / 4096.118164f)) - 1.f;

   // Flags from GBuffer:
   const bool is_vehicle_flag = ((gbuffer.w >> 13) & 1u) != 0u;
   const bool is_mirror_flag  = ((gbuffer.z & 1u) != 0u);
   const bool is_close_depth  = (depth < 2.0f);

   // Color Legend:
   // --------------------------------------------------------------------------
   // RED     (1, 0, 0) = Vehicle Flag (t15.w bit 13)
   // GREEN   (0, 1, 0) = Mirror / Dynamic Screen Flag (t15.z bit 0)
   // BLUE    (0, 0, 1) = Depth < 2.0m (close cockpit interior space)
   //
   // Combinations:
   // YELLOW  (1, 1, 0) = Vehicle AND Mirror Flag (Truck Mirror Glass / Screen)
   // MAGENTA (1, 0, 1) = Vehicle AND Depth < 2.0m (Cabin Interior)
   // CYAN    (0, 1, 1) = Mirror Flag AND Depth < 2.0m (Interior Screen)
   // WHITE   (1, 1, 1) = Vehicle AND Mirror AND Depth < 2.0m
   // BLACK   (0, 0, 0) = Outside World / Background (Road, Sky, Buildings)
   // --------------------------------------------------------------------------

   float3 mask_color = float3(0.0f, 0.0f, 0.0f);

   if (is_vehicle_flag) mask_color.r = 1.0f;
   if (is_mirror_flag)  mask_color.g = 1.0f;
   if (is_close_depth)  mask_color.b = 1.0f;

#if BLEND_OVER_GAME
   const float4 scene = SceneColor.Load(int3(pixel, 0));
   const float3 blended = lerp(scene.rgb, mask_color, 0.65f);
   return float4(blended, 1.0f);
#else
   return float4(mask_color, 1.0f);
#endif
}
