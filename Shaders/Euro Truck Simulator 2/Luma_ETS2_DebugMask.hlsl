// ETS2 DLAA: High-Contrast Debug Visual Mask
// Pure Black background for the outside world, high-contrast colors for cabin elements.
//
// Inputs:
//   t0 = SRV15, R16G16B16A16_UINT GBuffer flags written by geometry pass
//          .w [bit 13] vehicle / cockpit geometry flag
//          .z [bit 0]  frit / material flag
//   t1 = SRV4,  R16G16_UINT Packed Depth
//   t2 = SRV1,  RGBA16F scene color
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

   const bool flag_w13 = ((gbuffer.w >> 13) & 1u) != 0u; // Vehicle geometry flag
   const bool flag_z0  = ((gbuffer.z & 1u) != 0u);        // Frit / material flag

   // Outside world: strictly 100% pure solid black!
   if (!flag_w13)
   {
      return float4(0.0f, 0.0f, 0.0f, 1.0f);
   }

   // Inside vehicle:
   // - YELLOW: Frit wave-dots / glass border (vehicle + z0)
   // - RED: Pure vehicle interior / chassis / steering wheel / mirror
   if (flag_z0)
   {
      return float4(1.0f, 0.90f, 0.0f, 1.0f); // Bright Yellow: Frit dots
   }

   return float4(1.0f, 0.0f, 0.0f, 1.0f); // Bright Red: Vehicle body & interior
}
