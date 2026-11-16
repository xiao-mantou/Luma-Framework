// Euro Truck Simulator 2 DLAA - Mirror Reactive Mask
// Outputs solid 1.0f into the DLSS Reactive / Bias Current Color Mask (R8_UNORM)
// to eliminate temporal smearing on rearview mirrors and dynamic electronic screens.

float4 main() : SV_Target0
{
   return float4(1.0f, 1.0f, 1.0f, 1.0f);
}
