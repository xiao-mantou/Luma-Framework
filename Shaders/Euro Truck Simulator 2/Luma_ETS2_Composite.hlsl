// ETS2 DLAA: Hybrid Composition Shader
// Blends between the DLAA upscaled/anti-aliased image (outside world)
// and the de-jittered, razor-sharp native current frame color (cockpit interior).
//
// Inputs:
//   t0 = DLAA resolved colour (RGBA16F)
//   t1 = Native current frame colour (RGBA16F from slot 1)
//   t2 = GBuffer flags (SRV 15, R16G16B16A16_UINT)
//          .w [bit 13] vehicle / cockpit geometry flag
//   s0 = LinearSampler (hardware bilinear filtering for subpixel de-jittering)
//   b0 = GameCB0, where:
//          cb0[8].xy  carries engine camera projection jitter (UV space)
//          cb0[14].y  carries cabin_dlaa_blend (0.0 = 100% native, 1.0 = 100% DLAA)
//          cb0[14].zw carries C++ validated jitter_uv (UV space)

cbuffer GameCB0 : register(b0)
{
   float4 cb0[16];
};

Texture2D<float4> DlaaTexture : register(t0);
Texture2D<float4> NativeSourceTexture : register(t1);
Texture2D<uint4> GBufferFlags : register(t2);
SamplerState LinearSampler : register(s0);

float4 main(float4 pos : SV_Position) : SV_Target0
{
   const int2 pixel = int2(pos.xy);
   const uint4 gbuffer = GBufferFlags.Load(int3(pixel, 0));
   const bool is_vehicle = ((gbuffer.w >> 13) & 1u) != 0u;

   const float4 dlaa = DlaaTexture.Load(int3(pixel, 0));

   // Fast path: outside world pixels bypass interior checks completely
   if (!is_vehicle)
   {
      return dlaa;
   }

   // Texture dimensions for exact UV mapping
   float width, height;
   NativeSourceTexture.GetDimensions(width, height);
   const float2 render_size = float2(width, height);

   // Target pixel unjittered UV (pos.xy is at half-pixel center x+0.5, y+0.5)
   const float2 uv = pos.xy / render_size;

   // Projection jitter offset from C++ (cb0[14].zw) with fallback to cb0[8].xy
   const float2 jitter_uv = (cb0[14].z != 0.0f || cb0[14].w != 0.0f) ? cb0[14].zw : cb0[8].xy;

   // Subpixel de-jitter: sampling at (uv + jitter_uv) cancels the camera projection jitter!
   // Using hardware linear filtering (SampleLevel) guarantees smooth, exact subpixel interpolation.
   const float2 dejittered_uv = uv + jitter_uv;
   const float4 native_src = NativeSourceTexture.SampleLevel(LinearSampler, dejittered_uv, 0.0f);

   const float blend = saturate(cb0[14].y);
   return lerp(native_src, dlaa, blend);
}
