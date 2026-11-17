// Euro Truck Simulator 2 DLAA - Mirror & Dynamic Screen Reactive Mask
// Reads GBuffer flags (SRV 15, R16G16B16A16_UINT) written by geometry pass.
// Disassembly of ETS2 TAA resolve 0x805A8981 (lines 5-8, 67, 136, 262):
//   bit 0 of .z marks rearview mirror glass and dynamic electronic screens.
// When true, original TAA discards history and blends current frame with 100% weight.
// We write 1.0f into DLSS Reactive / Bias Current Color Mask (pInBiasCurrentColorMask)
// to eliminate temporal smear/ghosting inside rearview mirror reflections.

Texture2D<uint4> GBufferFlags : register(t0);

float main(float4 pos : SV_Position) : SV_Target0
{
   const uint4 gbuffer = GBufferFlags.Load(int3(pos.xy, 0));
   const bool is_mirror_glass = ((gbuffer.z & 1u) != 0u);
   return is_mirror_glass ? 1.0f : 0.0f;
}
