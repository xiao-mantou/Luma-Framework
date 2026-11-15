///////////////////////////////////////////////////////////////////////
// Euro Truck Simulator 2 - DLAA mod, v1 probe
//
// Pipeline facts from a RenoDX DevKit capture (ETS2 1.62.3.3, DX11 SM5X,
// 1920x1080, native BT.2100 PQ swapchain, in-game TAA enabled):
//   #6127 0xB824672E  packs neighbour min/max depth into R16G16_UINT
//                     (log quantised; NOT a float motion vector buffer)
//   #6128 0x805A8981  TAA temporal resolve. MRT 2x R16G16B16A16_FLOAT,
//                     SRVs: slot0/1 RGBA16F, slot3/4 R16G16_UINT,
//                     slot15 RGBA16_UINT; CB slot 0 (256B @ 116480)
//                     carries the per-frame jitter. <-- DLAA target
//   #6133..#6238      UI/HUD (B8G8R8(A)8_UNORM_SRGB) composited into the
//                     same RGBA16F target
//   #6239 0x7258FF32  BT.2020 matrix + ST2084 PQ encode ->
//                     R10G10B10A2_UNORM swapchain. Keep untouched.
//
// Colour input is linear HDR: 27.2% of pixels have a channel > 1.0 (max ~10).
// No HDR or swapchain work is needed; this mod only replaces TAA with DLAA.
//
// v1 proved Luma's shader hashing finds the TAA resolve (hits/frame ~= 1).
// v2 stage 1 isolates that resolve behind a debug toggle: cancelling it must
// produce an unresolved/jittered frame. v2 stage 2 wires DLAA once the packed
// uint motion vectors are unpacked and the main depth is captured.
///////////////////////////////////////////////////////////////////////

// Specify a define with the name of the game here (GAME_*).
// This is optional but can be used to hardcode custom (per game) behaviours in the core library,
// or other reasons. We can't automate it through the project define system (based on the project name) because
// we need these defines to be upper case and space free.
#define GAME_EURO_TRUCK_SIMULATOR_2 1

//// Define all the global "core" defines before including its files: ////

// Enable support for Geometry Shaders (GS), rather rare.
#define GEOMETRY_SHADER_SUPPORT 0

// Enable to stop "You can now attach the debugger" popup.
#define DISABLE_AUTO_DEBUGGER 0

// DLSS/FSR are toggled via the "Luma Properties" page in the project properties (UseLumaNGX/UseLumaFSR).

// Needed to run the engine's own TAA resolve before overwriting its colour RTV.
#define ENABLE_POST_DRAW_DISPATCH_CALLBACK 1

// Instead of "manually" including the "core" library, we simply include its main code file (which is a header).
// The library in itself is standalone, as in, it compiles fine and could directly be used as a template addon etc if built as dll but,
// there's a major limitation in how libraries dependencies work by nature, and that is that you can only make
// one version of them for all other projects to use. For performance and tidiness reasons, we are interested in
// having global defines that can be turned on and off per game, as opposed to runtime (static) parameters.
// Hence why we specify the global defines before including the core Luma file (where near all of the generic Luma implementation is).
// If we wanted to use a library, we'd also need to add a core "main" definition in a cpp file, to link it properly.
// All externs that are currently defined in core would also need to be manually defined in each game's implementation (e.g. see "RESHADE_EXTERNS" above).
// The only disadvantage of not actually including the core as a library, is that we'll have to add the same include/library dependencies in our
// game project (e.g. add ReShade, DLSS, etc), and manually add all cpp files too.
//
// To compile in different modes (e.g. "DEVELOPMENT", "TEST" etc see "global_defines.h").
#include "..\..\Core\core.hpp"

// PS hash of the TAA temporal resolve to be replaced by DLAA.
static constexpr uint32_t kTaaResolvePS = 0x805A8981u;
// LoadConfigs() has no DeviceData, so ini values are staged here and copied in OnCreateDevice().
static int g_cancel_taa_resolve_ini = 0;
static int g_enable_dlaa_ini = 0;
static int g_test_output_path_ini = 0;
static int g_bypass_ngx_ini = 0;
static int g_source_slot_ini = -1; // -1 = auto (pick first RGBA16F SRV that is not RTV0)
static int g_keep_engine_resolve_ini = 1;
static int g_mark_dlaa_output_ini = 0;

struct ETS2DeviceData final : public GameDeviceData
{
   bool has_drawn_taa_resolve = false;
   uint32_t taa_resolve_hits = 0; // cumulative, shown in the ImGui panel
   uint32_t frame_index = 0;
   bool enable_dlaa = false;          // v2 stage 2: real DLAA, still off
   bool cancel_taa_resolve = false;   // v2 stage 1 debug toggle
   bool dlaa_drew_this_frame = false; // reset every OnPresent; do NOT use core's has_drawn_sr
   bool test_output_path = false;     // debug: copy a solid red texture instead of running DLSS
#if ENABLE_SR
   bool bypass_ngx = false; // debug: copy the colour SRV straight into RTV0
   int source_slot = -1;    // -1 = auto
   int resolved_source_slot = 0;
   bool source_equals_output = false; // true => NGX would read back its own output
   bool keep_engine_resolve = true;   // run the original draw first, then overwrite RTV0
   bool original_draw_available = false;
   uint32_t next_stats_frame = 0;
   uint32_t stats_logs = 0;
   bool mark_dlaa_output = false; // debug: stamp a magenta block on our output
   uint32_t next_heartbeat_frame = 0;
   com_ptr<ID3D11Texture2D> zero_mv;        // placeholder: all-zero float2 motion
   com_ptr<ID3D11Texture2D> const_depth;    // placeholder: constant HW depth
   com_ptr<ID3D11Texture2D> uav_output;     // proxy when RTV0 lacks UAV
   com_ptr<ID3D11Texture2D> staging_source; // snapshot when source == output
   uint32_t sr_draws = 0;
   uint32_t sr_failures = 0;
   const char* sr_status = "off";
#endif
};

#if ENABLE_SR
// Stage 2 link validation. ETS2 has no standalone float motion vector buffer:
// #6127 packs neighbour min/max depth into R16G16_UINT. Until that is decoded,
// feed DLAA zero motion + a constant depth. Every failure path returns false so
// the engine keeps its own TAA; this function never leaves the frame broken.
static void SetSrStatus(ETS2DeviceData& gd, const char* status)
{
   if (gd.sr_status == status)
      return;
   gd.sr_status = status;
   const std::string message = std::string("[ETS2 DLAA] ") + status;
   reshade::log::message(reshade::log::level::warning, message.c_str());
}

static float HalfToFloat(uint16_t h)
{
   const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
   const uint32_t exp = (h >> 10) & 0x1Fu;
   const uint32_t mant = h & 0x03FFu;
   uint32_t bits;
   if (exp == 0)
      bits = sign; // subnormals are irrelevant for luminance statistics
   else if (exp == 31)
      bits = sign | 0x7F800000u | (mant << 13);
   else
      bits = sign | ((exp + 112u) << 23) | (mant << 13);
   union
   {
      uint32_t u;
      float f;
   } v = {bits};
   return v.f;
}

struct TexStats
{
   float mean = 0.f;
   float max = 0.f;
   float nonzero = 0.f;
   bool ok = false;
};

// One-shot CPU readback of a RGBA16F resource. Samples a 32x32 grid so this
// never stalls the frame for long.
static TexStats ReadTexStats(ID3D11Device* device, ID3D11DeviceContext* ctx, ID3D11Resource* res)
{
   TexStats st;
   if (!res)
      return st;
   com_ptr<ID3D11Texture2D> tex;
   if (FAILED(res->QueryInterface(&tex)) || !tex)
      return st;
   D3D11_TEXTURE2D_DESC d = {};
   tex->GetDesc(&d);
   if (d.Format != DXGI_FORMAT_R16G16B16A16_FLOAT || d.SampleDesc.Count != 1)
      return st;
   D3D11_TEXTURE2D_DESC sd = d;
   sd.Usage = D3D11_USAGE_STAGING;
   sd.BindFlags = 0;
   sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
   sd.MiscFlags = 0;
   com_ptr<ID3D11Texture2D> staging;
   if (FAILED(device->CreateTexture2D(&sd, nullptr, &staging)) || !staging)
      return st;
   ctx->CopyResource(staging.get(), res);
   D3D11_MAPPED_SUBRESOURCE m = {};
   if (FAILED(ctx->Map(staging.get(), 0, D3D11_MAP_READ, 0, &m)))
      return st;
   const UINT step_x = d.Width > 32 ? d.Width / 32 : 1;
   const UINT step_y = d.Height > 32 ? d.Height / 32 : 1;
   double sum = 0.0;
   float mx = 0.f;
   uint32_t n = 0;
   uint32_t nz = 0;
   for (UINT y = 0; y < d.Height; y += step_y)
   {
      const uint16_t* row = reinterpret_cast<const uint16_t*>(static_cast<const char*>(m.pData) + static_cast<size_t>(y) * m.RowPitch);
      for (UINT x = 0; x < d.Width; x += step_x)
      {
         const float r = HalfToFloat(row[x * 4 + 0]);
         const float g = HalfToFloat(row[x * 4 + 1]);
         const float b = HalfToFloat(row[x * 4 + 2]);
         const float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
         sum += lum;
         if (lum > mx)
            mx = lum;
         if (lum > 0.001f)
            ++nz;
         ++n;
      }
   }
   ctx->Unmap(staging.get(), 0);
   st.mean = n ? static_cast<float>(sum / n) : 0.f;
   st.max = mx;
   st.nonzero = n ? static_cast<float>(nz) / n : 0.f;
   st.ok = true;
   return st;
}

static std::string StatsText(const TexStats& st)
{
   if (!st.ok)
      return "n/a";
   char buf[128];
   sprintf_s(buf, "mean=%.3f max=%.3f nz=%.2f", st.mean, st.max, st.nonzero);
   return buf;
}

// Stamps a 128x128 magenta block into the top-left of the resource we are
// about to hand back to the game. Seeing it on screen proves our write landed.
static void ApplyDlaaMarker(ID3D11DeviceContext* ctx, ID3D11Resource* res, UINT w, UINT h)
{
   constexpr UINT kSize = 128;
   const UINT mw = w < kSize ? w : kSize;
   const UINT mh = h < kSize ? h : kSize;
   static std::vector<uint16_t> marker;
   if (marker.empty())
   {
      marker.assign(static_cast<size_t>(mw) * mh * 4, 0);
      for (size_t i = 0; i < static_cast<size_t>(mw) * mh; ++i)
      {
         marker[i * 4 + 0] = 0x3C00; // half 1.0 red
         marker[i * 4 + 2] = 0x3C00; // half 1.0 blue
         marker[i * 4 + 3] = 0x3C00; // alpha
      }
   }
   const D3D11_BOX box = {0, 0, 0, mw, mh, 1};
   ctx->UpdateSubresource(res, 0, &box, marker.data(), mw * 8, 0);
}

static bool TryDrawDLAA(ID3D11Device* device, ID3D11DeviceContext* ctx, DeviceData& device_data, ETS2DeviceData& gd)
{
   if (device_data.sr_type == SR::Type::None)
   {
      SetSrStatus(gd, "sr_type=None: pick DLSS in Luma > Super Resolution");
      return false;
   }
   if (device_data.sr_suppressed)
   {
      SetSrStatus(gd, "SR suppressed by core");
      return false;
   }
   if (gd.dlaa_drew_this_frame)
   {
      SetSrStatus(gd, "DLAA already drew this frame (duplicate resolve)");
      return false;
   }
   auto* instance = device_data.GetSRInstanceData();
   auto* impl = sr_implementations.count(device_data.sr_type) ? sr_implementations[device_data.sr_type].get() : nullptr;
   if (!instance || !impl)
   {
      SetSrStatus(gd, "no SR instance (pick DLSS in the Luma menu)");
      return false;
   }

   com_ptr<ID3D11ShaderResourceView> srvs[8] = {};
   com_ptr<ID3D11RenderTargetView> rtv;
   com_ptr<ID3D11DepthStencilView> dsv;
   ctx->PSGetShaderResources(0, 8, &srvs[0]);
   ctx->OMGetRenderTargets(1, &rtv, &dsv);

   com_ptr<ID3D11Resource> output_color;
   if (rtv)
      rtv->GetResource(&output_color);
   if (!output_color)
   {
      SetSrStatus(gd, "missing RTV0");
      return false;
   }

   // One-shot audit of every PS SRV slot. #6128 writes RTV0 (A15EA0), which is
   // also consumed as the next frame's history. If the colour input resolves to
   // that same resource, DLSS reads back its own output -> feedback black frame.
   com_ptr<ID3D11Resource> srv_res[8] = {};
   DXGI_FORMAT srv_fmt[8] = {};
   bool srv_same[8] = {};
   for (int i = 0; i < 8; ++i)
   {
      srv_fmt[i] = DXGI_FORMAT_UNKNOWN;
      if (!srvs[i])
         continue;
      D3D11_SHADER_RESOURCE_VIEW_DESC vd = {};
      srvs[i]->GetDesc(&vd);
      srv_fmt[i] = vd.Format;
      srvs[i]->GetResource(&srv_res[i]);
      srv_same[i] = srv_res[i] && srv_res[i].get() == output_color.get();
   }

   int slot = gd.source_slot;
   if (slot < 0 || slot > 7)
   {
      slot = 0;
      for (int i = 0; i < 8; ++i)
      {
         if (srv_res[i] && !srv_same[i] && srv_fmt[i] == DXGI_FORMAT_R16G16B16A16_FLOAT)
         {
            slot = i;
            break;
         }
      }
   }
   gd.resolved_source_slot = slot;
   if (!srvs[slot] || !srv_res[slot])
   {
      SetSrStatus(gd, "missing colour SRV");
      return false;
   }

   com_ptr<ID3D11Resource>& source_color = srv_res[slot];
   com_ptr<ID3D11Resource> depth_resource;
   if (dsv)
      dsv->GetResource(&depth_resource);
   const bool same_resource = srv_same[slot];
   gd.source_equals_output = same_resource;

   static bool logged_bindings = false;
   if (!logged_bindings)
   {
      logged_bindings = true;
      std::string line = "[ETS2 DLAA] bindings out=" +
                         std::to_string(reinterpret_cast<unsigned long long>(output_color.get())) +
                         " slot=" + std::to_string(slot);
      for (int i = 0; i < 8; ++i)
         line += " | s" + std::to_string(i) + "=" +
                 std::to_string(reinterpret_cast<unsigned long long>(srv_res[i].get())) +
                 " fmt=" + std::to_string(static_cast<int>(srv_fmt[i])) + (srv_same[i] ? " SAME" : "");
      reshade::log::message(reshade::log::level::warning, line.c_str());
   }

   // Periodic readback: a single frame can catch a cleared/ping-pong buffer.
   if (gd.frame_index >= gd.next_stats_frame && gd.stats_logs < 8)
   {
      gd.next_stats_frame = gd.frame_index + 60;
      ++gd.stats_logs;
      std::string stats = "[ETS2 DLAA] srv_stats f=" + std::to_string(gd.frame_index);
      for (int i = 0; i < 8; ++i)
      {
         if (!srv_res[i] || srv_fmt[i] != DXGI_FORMAT_R16G16B16A16_FLOAT)
            continue;
         stats += " s" + std::to_string(i) + " [" + StatsText(ReadTexStats(device, ctx, srv_res[i].get())) + "]";
      }
      // RTV0 is still bound to the OM here, so it is not a legal readback source.
      // Its contents are verified after the copy, once the OM has been detached.
      reshade::log::message(reshade::log::level::warning, stats.c_str());
   }

   com_ptr<ID3D11Texture2D> output_tex;
   if (FAILED(output_color->QueryInterface(&output_tex)) || !output_tex)
   {
      SetSrStatus(gd, "output is not a 2D texture");
      return false;
   }
   D3D11_TEXTURE2D_DESC out_desc = {};
   output_tex->GetDesc(&out_desc);
   const UINT w = out_desc.Width;
   const UINT h = out_desc.Height;

   // Debug: prove the output path itself. If the screen turns solid red, the
   // unbind/Copy/restore chain works and a black DLAA frame means NGX rejected
   // the placeholder inputs. If it stays dark, CopyResource is the culprit and
   // we need the Luma_Copy_PS full-screen pass instead.
   if (gd.test_output_path)
   {
      D3D11_TEXTURE2D_DESC d = out_desc;
      std::vector<uint16_t> pixels(static_cast<size_t>(w) * h * 4, 0);
      for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i)
      {
         pixels[i * 4 + 0] = 0x3C00; // half-float 1.0 (red)
         pixels[i * 4 + 3] = 0x3C00; // alpha
      }
      D3D11_SUBRESOURCE_DATA init = {pixels.data(), w * 8, 0};
      com_ptr<ID3D11Texture2D> red;
      if (SUCCEEDED(device->CreateTexture2D(&d, &init, &red)))
      {
         ID3D11RenderTargetView* saved_rtv = rtv.get();
         ID3D11DepthStencilView* saved_dsv = dsv.get();
         ID3D11RenderTargetView* null_rtv = nullptr;
         ctx->OMSetRenderTargets(1, &null_rtv, nullptr);
         ctx->CopyResource(output_color.get(), red.get());
         ctx->OMSetRenderTargets(1, &saved_rtv, saved_dsv);
         SetSrStatus(gd, "output path test: solid red");
         gd.dlaa_drew_this_frame = true;
         return true;
      }
      SetSrStatus(gd, "output path test: texture alloc failed");
      return false;
   }

   // Debug: bypass NGX entirely and push the selected colour SRV through the
   // exact same unbind/Copy/restore chain. A clean frame here proves the output
   // path and every binding are fine, so a black DLAA frame is NGX-side.
   if (gd.bypass_ngx)
   {
      ID3D11RenderTargetView* saved_rtv = rtv.get();
      ID3D11DepthStencilView* saved_dsv = dsv.get();
      ID3D11RenderTargetView* null_rtv = nullptr;
      // An SRV bound to the PS may keep the resource alive in a state where the
      // driver drops the copy. Detach all PS SRVs for the copy, then restore.
      com_ptr<ID3D11ShaderResourceView> saved_ps_srvs[16] = {};
      ctx->PSGetShaderResources(0, 16, &saved_ps_srvs[0]);
      ID3D11ShaderResourceView* null_srvs[16] = {};
      ctx->PSSetShaderResources(0, 16, null_srvs);
      ctx->OMSetRenderTargets(1, &null_rtv, nullptr);
      if (!same_resource)
         ctx->CopyResource(output_color.get(), source_color.get());
      if (gd.mark_dlaa_output)
         ApplyDlaaMarker(ctx, output_color.get(), w, h);
      static bool logged_bypass_copy = false;
      if (!logged_bypass_copy)
      {
         logged_bypass_copy = true;
         const std::string msg = "[ETS2 DLAA] bypass_copy src [" + StatsText(ReadTexStats(device, ctx, source_color.get())) +
                                 "] dst [" + StatsText(ReadTexStats(device, ctx, output_color.get())) + "]";
         reshade::log::message(reshade::log::level::warning, msg.c_str());
      }
      ID3D11ShaderResourceView* restore_srvs[16] = {};
      for (int i = 0; i < 16; ++i)
         restore_srvs[i] = saved_ps_srvs[i].get();
      ctx->PSSetShaderResources(0, 16, restore_srvs);
      ctx->OMSetRenderTargets(1, &saved_rtv, saved_dsv);
      SetSrStatus(gd, same_resource ? "bypass NGX: source IS output (no copy)" : "bypass NGX: colour SRV copied to RTV0");
      gd.dlaa_drew_this_frame = true;
      return true;
   }

   if (!gd.zero_mv || !gd.const_depth)
   {
      D3D11_TEXTURE2D_DESC d = {};
      d.Width = w;
      d.Height = h;
      d.MipLevels = 1;
      d.ArraySize = 1;
      d.SampleDesc.Count = 1;
      d.Usage = D3D11_USAGE_DEFAULT;
      d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      d.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
      std::vector<uint16_t> zeros(static_cast<size_t>(w) * h * 4, 0);
      D3D11_SUBRESOURCE_DATA init = {zeros.data(), w * 8, 0};
      device->CreateTexture2D(&d, &init, &gd.zero_mv);
      d.Format = DXGI_FORMAT_R32_FLOAT;
      std::vector<float> ones(static_cast<size_t>(w) * h, 1.f);
      init = {ones.data(), w * 4, 0};
      device->CreateTexture2D(&d, &init, &gd.const_depth);
   }
   if (!gd.zero_mv || !gd.const_depth)
   {
      SetSrStatus(gd, "placeholder texture alloc failed");
      return false;
   }

   SR::SettingsData settings = {};
   settings.output_width = w;
   settings.output_height = h;
   settings.render_width = w;
   settings.render_height = h;
   settings.hdr = true;             // ETS2 resolves in linear RGBA16F
   settings.inverted_depth = false; // TODO: verify ETS2 depth direction
   settings.mvs_jittered = false;
   settings.auto_exposure = false;
   settings.mvs_x_scale = 1.f;
   settings.mvs_y_scale = 1.f;
   settings.render_preset = dlss_render_preset;
   if (!impl->UpdateSettings(instance, ctx, settings))
   {
      SetSrStatus(gd, "UpdateSettings failed");
      device_data.force_reset_sr = true;
      return false;
   }

   ID3D11Resource* dlss_output = output_color.get();
   bool need_copy = false;
   if ((out_desc.BindFlags & D3D11_BIND_UNORDERED_ACCESS) == 0)
   {
      D3D11_TEXTURE2D_DESC uav_desc = out_desc;
      uav_desc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
      gd.uav_output = nullptr;
      if (FAILED(device->CreateTexture2D(&uav_desc, nullptr, &gd.uav_output)) || !gd.uav_output)
      {
         SetSrStatus(gd, "UAV proxy alloc failed");
         return false;
      }
      dlss_output = gd.uav_output.get();
      need_copy = true;
   }

   // One-shot branch diagnostic: straight UAV write into RTV0 vs. UAV proxy + CopyResource.
   static bool logged_branch = false;
   if (!logged_branch)
   {
      logged_branch = true;
      const std::string branch = "[ETS2 DLAA] branch need_copy=" + std::to_string(need_copy ? 1 : 0) +
                                 " bind_flags=" + std::to_string(out_desc.BindFlags) +
                                 " has_uav=" + std::to_string((out_desc.BindFlags & D3D11_BIND_UNORDERED_ACCESS) != 0 ? 1 : 0) +
                                 " size=" + std::to_string(w) + "x" + std::to_string(h) +
                                 " format=" + std::to_string(static_cast<int>(out_desc.Format)) +
                                 " depth=" + std::to_string(depth_resource ? 1 : 0);
      reshade::log::message(reshade::log::level::warning, branch.c_str());
   }

   // Never let NGX read the very resource it is about to overwrite.
   ID3D11Resource* dlss_source = source_color.get();
   if (same_resource)
   {
      if (!gd.staging_source)
      {
         // Keep size/format/mip identical to RTV0 so CopyResource stays legal.
         D3D11_TEXTURE2D_DESC sdesc = out_desc;
         sdesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
         device->CreateTexture2D(&sdesc, nullptr, &gd.staging_source);
      }
      if (gd.staging_source)
         dlss_source = gd.staging_source.get();
   }

   SR::SuperResolutionImpl::DrawData draw = {};
   draw.source_color = dlss_source;
   draw.output_color = dlss_output;
   draw.motion_vectors = gd.zero_mv.get();
   draw.depth_buffer = depth_resource ? depth_resource.get() : gd.const_depth.get();
   draw.render_width = w;
   draw.render_height = h;
   draw.jitter_x = 0.f; // TODO: read the real jitter from CB slot 0
   draw.jitter_y = 0.f;
   draw.reset = device_data.force_reset_sr;

   // NGX writes through a UAV while the same resource is still bound as the
   // current render target. D3D11 rejects that conflict, and CopyResource into
   // a bound RTV is illegal too. Unbind the OM for Draw + copy, then restore.
   ID3D11RenderTargetView* saved_rtv = rtv.get();
   ID3D11DepthStencilView* saved_dsv = dsv.get();
   ID3D11RenderTargetView* null_rtv = nullptr;
   ctx->OMSetRenderTargets(1, &null_rtv, nullptr);

   if (same_resource && gd.staging_source)
      ctx->CopyResource(gd.staging_source.get(), source_color.get());

   const bool draw_ok = impl->Draw(instance, ctx, draw);

   // DLSS is a compute pass and can leave its output UAV bound on return. A
   // resource still bound as UAV is not a valid CopyResource source on some
   // drivers, which would silently drop the copy (black frame). Detach the CS
   // UAVs for the copy and restore them afterwards.
   com_ptr<ID3D11UnorderedAccessView> bound_uavs[8] = {};
   ctx->CSGetUnorderedAccessViews(0, 8, &bound_uavs[0]);
   uint32_t bound_uav_count = 0;
   for (int i = 0; i < 8; ++i)
      bound_uav_count += bound_uavs[i] ? 1u : 0u;
   static bool logged_uav = false;
   if (!logged_uav)
   {
      logged_uav = true;
      reshade::log::message(reshade::log::level::warning,
         ("[ETS2 DLAA] cs_uavs_after_draw=" + std::to_string(bound_uav_count)).c_str());
   }
   ID3D11UnorderedAccessView* null_uavs[8] = {};
   ctx->CSSetUnorderedAccessViews(0, 8, null_uavs, nullptr);
   com_ptr<ID3D11ShaderResourceView> saved_ps_srvs[16] = {};
   ctx->PSGetShaderResources(0, 16, &saved_ps_srvs[0]);
   ID3D11ShaderResourceView* null_srvs[16] = {};
   ctx->PSSetShaderResources(0, 16, null_srvs);

   if (draw_ok && gd.mark_dlaa_output)
      ApplyDlaaMarker(ctx, dlss_output, w, h);
   if (draw_ok && need_copy)
      ctx->CopyResource(output_color.get(), dlss_output);

   ID3D11ShaderResourceView* restore_srvs[16] = {};
   for (int i = 0; i < 16; ++i)
      restore_srvs[i] = saved_ps_srvs[i].get();
   ctx->PSSetShaderResources(0, 16, restore_srvs);
   ID3D11UnorderedAccessView* restore_uavs[8] = {};
   for (int i = 0; i < 8; ++i)
      restore_uavs[i] = bound_uavs[i].get();
   ctx->CSSetUnorderedAccessViews(0, 8, restore_uavs, nullptr);
   ctx->OMSetRenderTargets(1, &saved_rtv, saved_dsv);

   static bool logged_dlss_copy = false;
   if (!logged_dlss_copy)
   {
      logged_dlss_copy = true;
      const std::string msg = "[ETS2 DLAA] dlss_copy out [" + StatsText(ReadTexStats(device, ctx, output_color.get())) +
                              "] uav [" + StatsText(ReadTexStats(device, ctx, dlss_output)) + "]";
      reshade::log::message(reshade::log::level::warning, msg.c_str());
   }

   if (!draw_ok)
   {
      SetSrStatus(gd, "Draw failed");
      return false;
   }

   SetSrStatus(gd, depth_resource ? "DLAA active (zero MV, real DSV)" : "DLAA active (zero MV, constant depth)");
   gd.dlaa_drew_this_frame = true;
   return true;
}
#endif

class ETS2Game final : public Game // Go view "Source/Core/includes/game.h" for all the overridable virtual functions
{
   // Optional helper to hide ugly casts
   static ETS2DeviceData& GetGameDeviceData(DeviceData& device_data)
   {
      return *static_cast<ETS2DeviceData*>(device_data.game);
   }

public:
   // You can define any data to initialize once here
   void OnInit(bool async) override
   {
      // You can add shader defines that will end up in the advanced settings for users to modify here.
      // These will be defined in the shaders, so they can be used to do static branches in them.
      // Ideally they should also be defined in the game's Settings.hlsl file.
      std::vector<ShaderDefineData> game_shader_defines_data = {
         {"TONEMAP_TYPE", /*default value*/ '1', true, false, /*tooltip*/ "0 - Vanilla SDR\n1 - Luma HDR (Vanilla+)", /*max value*/ 1},
      };
      shader_defines_data.append_range(game_shader_defines_data);
      assert(shader_defines_data.size() < MAX_SHADER_DEFINES); // Make sure there's room for at least one extra custom define to add for development (this isn't really relevant outside of development)

      // Define these according to the game's original technical details and the mod's implementation (see their declarations for more).
      GetShaderDefineData(POST_PROCESS_SPACE_TYPE_HASH).SetDefaultValue('0'); // What space are the colors in? Was the swapchain linear (sRGB texture format)? Did we change post processing to store in linear space?
      GetShaderDefineData(EARLY_DISPLAY_ENCODING_HASH).SetDefaultValue('0');  // Whether we do gamma correction and paper white scaling during post processing or we delay them until the final display composition pass
      GetShaderDefineData(VANILLA_ENCODING_TYPE_HASH).SetDefaultValue('0');   // What SDR transfer curve was the game using? Most modern games used sRGB in SDR
      GetShaderDefineData(GAMMA_CORRECTION_TYPE_HASH).SetDefaultValue('1');   // What SDR transfer curve to we want to emulate? This is relevant even if we work in linear space, as there can be a gamma mismatch on it
      GetShaderDefineData(UI_DRAW_TYPE_HASH).SetDefaultValue('0');            // How does the UI draw in?

      // Customize the cbuffers indexes here. They should be between 0 and 13 ("D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT - 1"),
      // set them to invalid values (e.g. -1) to not set them (they won't be uploaded to the GPU, and thus not usable in shaders). All their values need to be different if they are valid.
      // These will be automatically sent to the GPU for every shader pass the mod overrides.
      luma_settings_cbuffer_index = 13;
      luma_data_cbuffer_index = 12; // Needed for debugging textures and having custom per pass data (including handling the final pass properly).
      luma_ui_cbuffer_index = -1;   // Optional, see "UI_DRAW_TYPE" (this is for type 1)

      // Init the game settings here, you could also load them from ini in "LoadConfigs()".
      // These can be defined in a header shared between shaders and c++, and are sent to the GPU every time they change.
      //
      // It should be in "Shaders/Game/Includes/GameCBuffers.hlsl".
      //
      // To include it, the easiest way is adding as Forced Include within the Project Properties.
      // C/C++ -> All Options -> Forced Include File --> "../../../Shaders/$(ProjectName)/Includes/GameCBuffers.hlsl;" (w/o quotation marks)
      //
      // Otherwise, manually add it to the project files and include that at the top (before anything else, as the game cb settings struct needs to be defined),
      // or make a copy of it in c++ (making sure it's mirrored to hlsl), and define "LUMA_GAME_CB_STRUCTS" to avoid it being re-defined.
      //
      // Set "device_data.cb_luma_global_settings_dirty" to true if you changed values and need a re-upload to the GPU before OnPresent() (i.e. during OnDrawOrDispatch()).
      default_luma_global_game_settings.GameSetting01 = cb_luma_global_settings.GameSettings.GameSetting01 = 0.5f;
      default_luma_global_game_settings.GameSetting02 = cb_luma_global_settings.GameSettings.GameSetting02 = 33;
   }

   // This needs to be overridden with your own "GameDeviceData" sub-class (destruction is automatically handled)
   void OnCreateDevice(ID3D11Device* native_device, DeviceData& device_data) override
   {
      device_data.game = new ETS2DeviceData;
      auto* gd = static_cast<ETS2DeviceData*>(device_data.game);
      gd->cancel_taa_resolve = g_cancel_taa_resolve_ini != 0;
      gd->enable_dlaa = g_enable_dlaa_ini != 0;
      gd->test_output_path = g_test_output_path_ini != 0;
#if ENABLE_SR
      gd->bypass_ngx = g_bypass_ngx_ini != 0;
      gd->source_slot = g_source_slot_ini;
      gd->keep_engine_resolve = g_keep_engine_resolve_ini != 0;
      gd->mark_dlaa_output = g_mark_dlaa_output_ini != 0;
#endif
   }

   DrawOrDispatchOverrideType OnDrawOrDispatch(ID3D11Device* native_device, ID3D11DeviceContext* native_device_context, CommandListData& cmd_list_data, DeviceData& device_data, reshade::api::shader_stage stages, const ShaderHashesList<OneShaderPerPipeline>& original_shader_hashes, bool is_custom_pass, bool& updated_cbuffers, std::function<void()>* original_draw_dispatch_func) override
   {
      auto& game_device_data = GetGameDeviceData(device_data);

      // v1 probe: verify Luma's shader hash matches the DevKit hash of the TAA
      // resolve. If this counter stays at 0, the hash needs re-dumping with a
      // Development build before any DLAA work can be validated.
      if (original_shader_hashes.Contains(kTaaResolvePS, reshade::api::shader_stage::pixel))
      {
         ++game_device_data.taa_resolve_hits;
         game_device_data.has_drawn_taa_resolve = true;

         // v2 stage 1: prove this resolve is the sole writer of the final colour.
         // With the toggle on the frame should turn into an un-resolved, jittered
         // mess. If it stays clean, a later pass re-resolves over the output and
         // stage 2 will have to cancel that one too.
         if (game_device_data.cancel_taa_resolve)
            return DrawOrDispatchOverrideType::Skip;
#if ENABLE_SR
         if (game_device_data.enable_dlaa)
         {
            game_device_data.original_draw_available = original_draw_dispatch_func && *original_draw_dispatch_func;
            bool ran_original = false;
            if (game_device_data.keep_engine_resolve && game_device_data.original_draw_available)
            {
               // #6128 writes MRT 2 targets. Keep the engine's history/aux target
               // intact, then replace only RTV0 with the DLAA result.
               (*original_draw_dispatch_func)();
               ran_original = true;
            }
            if (TryDrawDLAA(native_device, native_device_context, device_data, game_device_data))
            {
               ++game_device_data.sr_draws;
               return DrawOrDispatchOverrideType::Replaced;
            }
            ++game_device_data.sr_failures;
            // Never let core issue the draw a second time.
            if (ran_original)
               return DrawOrDispatchOverrideType::Replaced;
         }
         if (game_device_data.frame_index >= game_device_data.next_heartbeat_frame)
         {
            game_device_data.next_heartbeat_frame = game_device_data.frame_index + 120;
            const std::string hb = "[ETS2 DLAA] heartbeat f=" + std::to_string(game_device_data.frame_index) +
                                   " hits=" + std::to_string(game_device_data.taa_resolve_hits) +
                                   " draws=" + std::to_string(game_device_data.sr_draws) +
                                   " fails=" + std::to_string(game_device_data.sr_failures) +
                                   " cb=" + std::to_string(game_device_data.original_draw_available ? 1 : 0) +
                                   " keep=" + std::to_string(game_device_data.keep_engine_resolve ? 1 : 0) +
                                   " slot=" + std::to_string(game_device_data.resolved_source_slot) +
                                   " same=" + std::to_string(game_device_data.source_equals_output ? 1 : 0);
            reshade::log::message(reshade::log::level::warning, hb.c_str());
         }
#endif
      }

      return DrawOrDispatchOverrideType::None;
   }
   void OnPresent(ID3D11Device* native_device, DeviceData& device_data) override
   {
      auto& game_device_data = GetGameDeviceData(device_data);

      game_device_data.has_drawn_taa_resolve = false;
      game_device_data.dlaa_drew_this_frame = false;
      ++game_device_data.frame_index;
   }

   void LoadConfigs() override
   {
      reshade::api::effect_runtime* runtime = nullptr;

      // ReShade.ini is ready here, so load its values.
      reshade::get_config_value(runtime, NAME, "GameSetting01", cb_luma_global_settings.GameSettings.GameSetting01);
      reshade::get_config_value(runtime, NAME, "GameSetting02", cb_luma_global_settings.GameSettings.GameSetting02);
      reshade::get_config_value(runtime, NAME, "CancelTaaResolve", g_cancel_taa_resolve_ini);
      reshade::get_config_value(runtime, NAME, "EnableDlaa", g_enable_dlaa_ini);
      reshade::get_config_value(runtime, NAME, "TestOutputPath", g_test_output_path_ini);
      reshade::get_config_value(runtime, NAME, "BypassNgx", g_bypass_ngx_ini);
      reshade::get_config_value(runtime, NAME, "SourceSlot", g_source_slot_ini);
      reshade::get_config_value(runtime, NAME, "KeepEngineResolve", g_keep_engine_resolve_ini);
      reshade::get_config_value(runtime, NAME, "MarkDlaaOutput", g_mark_dlaa_output_ini);
   }

   void DrawImGuiSettings(DeviceData& device_data) override
   {
      reshade::api::effect_runtime* runtime = nullptr;
      auto& game_device_data = GetGameDeviceData(device_data);

      ImGui::Text("ETS2 DLAA - v2 stage 1 (resolve isolation)");
      ImGui::Text("TAA resolve 0x805A8981 hits: %u (frame %u)", game_device_data.taa_resolve_hits, game_device_data.frame_index);
      if (game_device_data.frame_index > 0)
         ImGui::Text("hits/frame: %.2f", float(game_device_data.taa_resolve_hits) / float(game_device_data.frame_index));
#if ENABLE_SR
      ImGui::Text("SR type: %d (DLSS = %d)", static_cast<int>(device_data.sr_type), static_cast<int>(SR::Type::DLSS));
      if (ImGui::Checkbox("Enable DLAA (stage 2, placeholder MV/depth)", &game_device_data.enable_dlaa))
      {
         int value = game_device_data.enable_dlaa ? 1 : 0;
         reshade::set_config_value(runtime, NAME, "EnableDlaa", value);
      }
      ImGui::Text("DLAA draws: %u  failures: %u", game_device_data.sr_draws, game_device_data.sr_failures);
      ImGui::Text("status: %s", game_device_data.sr_status);
      if (ImGui::Checkbox("Test output path (solid red, skips DLSS)", &game_device_data.test_output_path))
      {
         int value = game_device_data.test_output_path ? 1 : 0;
         reshade::set_config_value(runtime, NAME, "TestOutputPath", value);
      }
      if (ImGui::Checkbox("Bypass NGX (copy colour SRV to RTV0)", &game_device_data.bypass_ngx))
      {
         int value = game_device_data.bypass_ngx ? 1 : 0;
         reshade::set_config_value(runtime, NAME, "BypassNgx", value);
      }
      if (ImGui::Checkbox("Keep engine TAA resolve (then overwrite RTV0)", &game_device_data.keep_engine_resolve))
      {
         int value = game_device_data.keep_engine_resolve ? 1 : 0;
         reshade::set_config_value(runtime, NAME, "KeepEngineResolve", value);
      }
      ImGui::Text("original draw callback: %s", game_device_data.original_draw_available ? "available" : "MISSING");
      if (ImGui::Checkbox("Mark our output (magenta block, proves overwrite)", &game_device_data.mark_dlaa_output))
      {
         int value = game_device_data.mark_dlaa_output ? 1 : 0;
         reshade::set_config_value(runtime, NAME, "MarkDlaaOutput", value);
      }
      if (ImGui::SliderInt("Colour SRV slot (-1=auto)", &game_device_data.source_slot, -1, 7))
         reshade::set_config_value(runtime, NAME, "SourceSlot", game_device_data.source_slot);
      ImGui::Text("resolved slot=%d source==output=%d", game_device_data.resolved_source_slot, game_device_data.source_equals_output ? 1 : 0);
      ImGui::Text("sr_suppressed=%d core_has_drawn_sr=%d dlaa_this_frame=%d", device_data.sr_suppressed ? 1 : 0, device_data.has_drawn_sr ? 1 : 0, game_device_data.dlaa_drew_this_frame ? 1 : 0);
      ImGui::Text("Select 'DLSS' in the Super Resolution combo above first.");
#endif
      if (ImGui::Checkbox("Cancel TAA resolve (debug)", &game_device_data.cancel_taa_resolve))
      {
         int value = game_device_data.cancel_taa_resolve ? 1 : 0;
         reshade::set_config_value(runtime, NAME, "CancelTaaResolve", value);
      }
      ImGui::Text("ON -> unresolved/jittered frame; clean frame means a later pass re-resolves.");
      ImGui::Separator();

      // Draw whatever ImGui to let users modify settings.
      if (ImGui::SliderFloat("Game Setting #01", &cb_luma_global_settings.GameSettings.GameSetting01, 0.f, 1.f, "%.3f"))
         reshade::set_config_value(runtime, NAME, "GameSetting01", cb_luma_global_settings.GameSettings.GameSetting01);
      if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
         ImGui::SetTooltip("Hi! This changes Game Setting #01.");
      DrawResetButton(cb_luma_global_settings.GameSettings.GameSetting01, default_luma_global_game_settings.GameSetting01, "GameSetting01", runtime);

      uint GameSetting02_min = 0;
      uint GameSetting02_max = 50;
      if (ImGui::SliderScalar("Game Setting #02", ImGuiDataType_U32, &cb_luma_global_settings.GameSettings.GameSetting02, &GameSetting02_min, &GameSetting02_max, "%u"))
         reshade::set_config_value(runtime, NAME, "GameSetting02", cb_luma_global_settings.GameSettings.GameSetting02);
      if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
         ImGui::SetTooltip("Hi! This changes Game Setting #02.");
      DrawResetButton(cb_luma_global_settings.GameSettings.GameSetting02, default_luma_global_game_settings.GameSetting02, "GameSetting02", runtime);
   }

   void PrintImGuiAbout() override
   {
      // Remember to credit Luma developers, the game mod creators, and all third party code that is used (plus, optionally, testers too)
      ImGui::Text("ETS2 DLAA - replaces the TAA resolve at PS 0x805A8981", "");
   }
};

// This is where everything starts from, the very first call to the dll.
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved)
{
   if (ul_reason_for_call == DLL_PROCESS_ATTACH)
   {
      // Setup the globals (e.g. name etc). It's good to do this stuff before registering the ReShade addon, to make sure the names are up to date.
      const char* project_name = PROJECT_NAME;
      const char* cleared_project_name = (project_name[0] == '_') ? (project_name + 1) : project_name; // Remove the potential "_" at the beginning. This can include spaces!

      uint32_t mod_version = 1; // Increase this to reset the game settings and shader binaries after making large changes to your mod
                                // NOTE: the website argument must never be nullptr; SetGlobals() copies it with
      // strncpy() (globals.h), and a null source crashes DllMain with error 1114.
      Globals::SetGlobals(cleared_project_name, "ETS2 DLAA v2 stage 2", "", mod_version);

      // The following can be toggled in the dev settings (it generally only fully applies after changing the game's resolution).
      // Default these to "false" for the mod to not do "anything" that might cause issues with the game by default.
      // ETS2 already outputs native BT.2100 PQ HDR. Do not touch the swapchain or
      // upgrade textures: keep the vanilla pipeline until DLAA is wired in.
      swapchain_format_upgrade_type = TextureFormatUpgradesType::None;
      swapchain_upgrade_type = SwapchainUpgradeType::None;
      texture_format_upgrades_type = TextureFormatUpgradesType::None;
      // Critical for a DLAA-only mod: core.hpp:5219 otherwise runs its
      // "Display Composition" pass whenever gamma correction / re-encoding /
      // gamut mapping is deemed necessary, which re-grades the whole image.
      // ETS2 already outputs native BT.2100 PQ, so leave its colour untouched.
      // This is the same setting Mafia Definitive Edition (DLAA only) uses.
      force_disable_display_composition = true;
      // Texture upgrades (8 bit unorm and 11 bit float etc to 16 bit float)
      texture_upgrade_formats = {
         reshade::api::format::r8g8b8a8_unorm,
         reshade::api::format::r8g8b8a8_unorm_srgb,
         reshade::api::format::r8g8b8a8_typeless,
         reshade::api::format::r8g8b8x8_unorm,
         reshade::api::format::r8g8b8x8_unorm_srgb,
         reshade::api::format::b8g8r8a8_unorm,
         reshade::api::format::b8g8r8a8_unorm_srgb,
         reshade::api::format::b8g8r8a8_typeless,
         reshade::api::format::b8g8r8x8_unorm,
         reshade::api::format::b8g8r8x8_unorm_srgb,
         reshade::api::format::b8g8r8x8_typeless,

         reshade::api::format::r10g10b10a2_unorm,
         reshade::api::format::r10g10b10a2_typeless,

         // reshade::api::format::r16g16b16a16_unorm,

         reshade::api::format::r11g11b10_float,
      };
      // Set this to the size and type of the main color grading LUT you might want to upgrade (optional)
      texture_format_upgrades_lut_size = 32;
      texture_format_upgrades_lut_dimensions = LUTDimensions::_2D;
      // In case the textures failed to upgrade, tweak the filtering conditions to be more lenient (e.g. aspect ratio checks etc).
      texture_format_upgrades_2d_size_filters = 0 | (uint32_t)TextureFormatUpgrades2DSizeFilters::SwapchainResolution | (uint32_t)TextureFormatUpgrades2DSizeFilters::SwapchainAspectRatio;

      // Update samples to override the bias, based on the rendering resolution etc
      enable_samplers_upgrade = false;

#if DEVELOPMENT // If you want to track any shader names over time, you can hardcode them here by hash (they can be a useful reference in the pipeline)
      forced_shader_names.emplace(std::stoul("805A8981", nullptr, 16), "ETS2 TAA resolve (DLAA target)");
#endif

#if !DEVELOPMENT // Put shaders that a previous version of the mod used but has ever since been deleted here, so that users updating the mod from an older version won't accidentally load them
      old_shader_file_names.emplace("Bloom_0xDC9373A8.ps_5_0.hlsl");
#endif

      // Create your game sub-class instance (it will be automatically destroyed on exit).
      // You do not need to do this if you have no custom data to store.
      game = new ETS2Game();
   }

   CoreMain(hModule, ul_reason_for_call, lpReserved);

   return TRUE;
}
