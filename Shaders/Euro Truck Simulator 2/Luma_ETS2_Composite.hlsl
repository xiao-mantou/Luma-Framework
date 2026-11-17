// ETS2 DLAA: Hybrid Composition Shader
// Blends between the DLAA upscaled/anti-aliased image (outside world)
// and the razor-sharp native current frame color (cockpit interior).
//
// Inputs:
//   t0 = DLAA resolved colour (RGBA16F)
//   t1 = Native current frame colour (RGBA16F from slot 0)
//   t2 = GBuffer flags (SRV 15, R16G16B16A16_UINT)
//          .w [bit 13] vehicle / cockpit geometry flag
//   b0 = GameCB0, where cb0[14].y carries cabin_dlaa_blend (0.0 = 100% native, 1.0 = 100% DLAA)

cbuffer GameCB0 : register(b0)
{
   float4 cb0[16];
};

Texture2D<float4> DlaaTexture : register(t0);
Texture2D<float4> NativeSourceTexture : register(t1);
Texture2D<uint4> GBufferFlags : register(t2);

float4 main(float4 pos : SV_Position) : SV_Target0
{
   const int2 pixel = int2(pos.xy);
   const uint4 gbuffer = GBufferFlags.Load(int3(pixel, 0));
   const bool is_vehicle = ((gbuffer.w >> 13) & 1u) != 0u;

   const float4 dlaa = DlaaTexture.Load(int3(pixel, 0));

   if (!is_vehicle)
   {
      return dlaa;
   }

   const float4 native_src = NativeSourceTexture.Load(int3(pixel, 0));
   const float blend = saturate(cb0[14].y); // default 0.0f = pure sharp native color

   return lerp(native_src, dlaa, blend);
}
