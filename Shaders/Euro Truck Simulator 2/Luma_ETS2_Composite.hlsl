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

   // Fast path: outside world pixels bypass interior checks completely
   if (!is_vehicle)
   {
      return dlaa;
   }

   // Boundary & Glass Frit (波点) detection:
   // Sample 4 cardinal neighbours to detect transitions between vehicle and window/outside.
   const uint v_r = (GBufferFlags.Load(int3(pixel + int2( 1,  0), 0)).w >> 13) & 1u;
   const uint v_l = (GBufferFlags.Load(int3(pixel + int2(-1,  0), 0)).w >> 13) & 1u;
   const uint v_d = (GBufferFlags.Load(int3(pixel + int2( 0,  1), 0)).w >> 13) & 1u;
   const uint v_u = (GBufferFlags.Load(int3(pixel + int2( 0, -1), 0)).w >> 13) & 1u;
   const uint v_sum = v_r + v_l + v_d + v_u;

   // If on boundary/frit dots (v_sum < 4), apply DLAA to eliminate harsh stippling and staircase jaggies.
   // Deep inside the cabin (v_sum == 4), edge_dlaa is 0.0, strictly following user blend.
   const float edge_dlaa = float(4u - v_sum) * 0.25f;
   const float blend = saturate(cb0[14].y);
   const float effective_blend = max(blend, edge_dlaa);

   const float4 native_src = NativeSourceTexture.Load(int3(pixel, 0));
   return lerp(native_src, dlaa, effective_blend);
}
