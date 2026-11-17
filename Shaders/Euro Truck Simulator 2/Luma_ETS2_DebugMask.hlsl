// ETS2 DLAA: Debug Visual Mask for inspecting GBuffer Flags, Depth, and Rearview Mirror
//
// Inputs:
//   t0 = SRV15, R16G16B16A16_UINT GBuffer flags written by geometry pass
//          .w [bit 13] vehicle / cockpit geometry flag
//          .w [bit 14..15] secondary vehicle / dynamic bits
//          .z [bit 0]  material flag / dynamic surface (t15.z & 1)
//          .z [bit 1]  secondary material flag (t15.z & 2)
//   t1 = SRV4,  R16G16_UINT Packed Depth
//   t2 = SRV1,  RGBA16F live scene color (for semi-transparent overlay)
//   b0 = GameCB0 (16 x float4)

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

   const bool flag_w13 = ((gbuffer.w >> 13) & 1u) != 0u; // Standard vehicle flag
   const bool flag_z0  = ((gbuffer.z & 1u) != 0u);        // Material flag / dynamic flag
   const bool flag_w14 = ((gbuffer.w >> 14) & 1u) != 0u; // Secondary w flag
   const bool flag_z1  = ((gbuffer.z & 2u) != 0u);        // Secondary z flag

   // Live scene underlay
   const float4 scene = SceneColor.Load(int3(pixel, 0));

   // Color encoding:
   //   - RED: Standard Vehicle interior (w13 == 1, z0 == 0)
   //   - YELLOW / GOLD: Vehicle with material flag z0 (w13 == 1, z0 == 1)
   //   - GREEN: Pure Material flag without vehicle (w13 == 0, z0 == 1)
   //   - CYAN: Secondary w14 flag
   //   - MAGENTA: Secondary z1 flag
   //   - UNMODIFIED SCENE: Neither flag is set (e.g. outside world, sky, forward transparent glass)
   float3 mask_color = float3(0.0f, 0.0f, 0.0f);
   bool has_highlight = false;

   if (flag_w13 && !flag_z0)
   {
      mask_color = float3(1.0f, 0.0f, 0.0f); // RED: Cabin interior
      has_highlight = true;
   }
   else if (flag_w13 && flag_z0)
   {
      mask_color = float3(1.0f, 0.85f, 0.0f); // YELLOW / GOLD: Vehicle with material flag z0
      has_highlight = true;
   }
   else if (!flag_w13 && flag_z0)
   {
      mask_color = float3(0.0f, 1.0f, 0.0f); // GREEN: Material flag z0 only
      has_highlight = true;
   }
   else if (flag_w14)
   {
      mask_color = float3(0.0f, 1.0f, 1.0f); // CYAN: w14
      has_highlight = true;
   }
   else if (flag_z1)
   {
      mask_color = float3(1.0f, 0.0f, 1.0f); // MAGENTA: z1
      has_highlight = true;
   }

   if (has_highlight)
   {
      // 55% opacity blend over live game scene so user can see what mesh is being highlighted
      return float4(lerp(scene.rgb, mask_color, 0.55f), 1.0f);
   }

   // No flags: show original game scene
   return scene;
}
