// Euro Truck Simulator 2 DLAA - Cockpit & Mirror Reactive Bias Mask
// Feeds DLSS pInBiasCurrentColorMask to eliminate ghosting and smearing.
//
// GBuffer flags (SRV 15, R16G16B16A16_UINT):
//   bit 13 of .w flags vehicle / cockpit interior geometry
//
// Constant Buffer b0:
//   cb0[14].x carries the user-tunable Cabin Bias Mask from the Luma ImGui panel (default 0.75f)

cbuffer GameCB0 : register(b0)
{
   float4 cb0[16];
};

Texture2D<uint4> GBufferFlags : register(t0);

float main(float4 pos : SV_Position) : SV_Target0
{
   const int2 pixel = int2(pos.xy);
   const uint4 gbuffer = GBufferFlags.Load(int3(pixel, 0));

   // Bit 13 of .w flags vehicle interior geometry (verified by user via Debug Mask)
   const bool is_vehicle = ((gbuffer.w >> 13) & 1u) != 0u;

   if (!is_vehicle)
      return 0.0f;

   // Read tunable bias value from cb0[14].x if non-zero, otherwise default to 0.75f
   const float cabin_bias = (cb0[14].x > 0.0f) ? cb0[14].x : 0.75f;

   return cabin_bias;
}
