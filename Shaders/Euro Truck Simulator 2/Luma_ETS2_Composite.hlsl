// ETS2 DLAA: Hybrid Composition Shader with 3x3 YCoCg Variance Clamping
// Blends between the DLAA upscaled/anti-aliased image (outside world)
// and the de-jittered, razor-sharp native current frame color (cockpit interior),
// with 3x3 neighborhood AABB variance clamping to completely eliminate motion ghosting.
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

// Fast, lossless RGB <-> YCoCg transforms (Luma standard)
float3 rgb_to_ycocg(float3 color)
{
   const float y  = dot(color, float3( 0.25f,  0.50f,  0.25f));
   const float co = dot(color, float3( 0.50f,  0.00f, -0.50f));
   const float cg = dot(color, float3(-0.25f,  0.50f, -0.25f));
   return float3(y, co, cg);
}

float3 ycocg_to_rgb(float3 color)
{
   const float r = dot(color, float3(1.0f,  1.0f, -1.0f));
   const float g = dot(color, float3(1.0f,  0.0f,  1.0f));
   const float b = dot(color, float3(1.0f, -1.0f, -1.0f));
   return float3(r, g, b);
}

// Intersects the color ray from AABB center with the box faces (Karis 2014 / Luma)
float3 clip_to_aabb(float3 color, float3 minc, float3 maxc)
{
   const float3 center = (minc + maxc) * 0.5f;
   const float3 extent = (maxc - minc) * 0.5f + 1e-4f;
   const float3 offset = color - center;
   const float3 units = abs(offset / extent);
   const float max_unit = max(max(units.x, units.y), max(units.z, 1.0f));
   return center + offset / max_unit;
}

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
   const float2 texel = 1.0f / render_size;

   // Target pixel unjittered UV (pos.xy is at half-pixel center x+0.5, y+0.5)
   const float2 uv = pos.xy * texel;

   // Projection jitter offset from C++ (cb0[14].zw) with fallback to cb0[8].xy
   const float2 jitter_uv = (cb0[14].z != 0.0f || cb0[14].w != 0.0f) ? cb0[14].zw : cb0[8].xy;

   // Base de-jittered UV coordinate for the center pixel
   const float2 dejittered_uv = uv + jitter_uv;

   // Fast exit if user set blend strictly to 0.0 (pure de-jittered native frame)
   const float blend = saturate(cb0[14].y);
   const float4 native_center = NativeSourceTexture.SampleLevel(LinearSampler, dejittered_uv, 0.0f);
   if (blend <= 1e-4f)
   {
      return native_center;
   }

   // 3x3 Neighborhood Gathering around de-jittered center:
   // Ensures the statistical color box itself contains zero camera subpixel jitter.
   const float2 offsets[9] = {
      float2(-1.0f, -1.0f), float2( 0.0f, -1.0f), float2( 1.0f, -1.0f),
      float2(-1.0f,  0.0f), float2( 0.0f,  0.0f), float2( 1.0f,  0.0f),
      float2(-1.0f,  1.0f), float2( 0.0f,  1.0f), float2( 1.0f,  1.0f)
   };

   float3 m1 = float3(0.0f, 0.0f, 0.0f);
   float3 m2 = float3(0.0f, 0.0f, 0.0f);
   float3 min_c = float3(1e9f, 1e9f, 1e9f);
   float3 max_c = float3(-1e9f, -1e9f, -1e9f);

   [unroll]
   for (int i = 0; i < 9; ++i)
   {
      float3 c = (i == 4) ? native_center.rgb : NativeSourceTexture.SampleLevel(LinearSampler, dejittered_uv + offsets[i] * texel, 0.0f).rgb;
      float3 y = rgb_to_ycocg(c);
      m1 += y;
      m2 += y * y;
      min_c = min(min_c, y);
      max_c = max(max_c, y);
   }

   m1 *= (1.0f / 9.0f);
   m2 *= (1.0f / 9.0f);
   const float3 sigma = sqrt(max(0.0f, m2 - m1 * m1));

   // Karis 2014 / UE4 industry standard variance box (gamma = 1.25),
   // strictly clamped within the neighborhood min/max to preserve micro-highlights.
   const float gamma = 1.25f;
   const float3 box_min = max(min_c, m1 - gamma * sigma);
   const float3 box_max = min(max_c, m1 + gamma * sigma);

   // Clip DLAA color to local 3x3 variance box in YCoCg space
   const float3 dlaa_ycocg = rgb_to_ycocg(dlaa.rgb);
   const float3 clamped_ycocg = clip_to_aabb(dlaa_ycocg, box_min, box_max);
   const float3 clamped_dlaa = ycocg_to_rgb(clamped_ycocg);

   // Output: smooth blend between pure knife-sharp native center and clamped anti-ghosting DLAA
   const float3 final_rgb = lerp(native_center.rgb, clamped_dlaa, blend);
   return float4(final_rgb, native_center.a);
}
