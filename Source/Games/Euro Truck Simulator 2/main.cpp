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

struct ETS2DeviceData final : public GameDeviceData
{
   bool has_drawn_taa_resolve = false;
   uint32_t taa_resolve_hits = 0; // cumulative, shown in the ImGui panel
   uint32_t frame_index = 0;
   bool enable_dlaa = false;        // v2 stage 2: real DLAA, still off
   bool cancel_taa_resolve = false; // v2 stage 1 debug toggle
};

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
      static_cast<ETS2DeviceData*>(device_data.game)->cancel_taa_resolve = g_cancel_taa_resolve_ini != 0;
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
      }

      return DrawOrDispatchOverrideType::None;
   }
   void OnPresent(ID3D11Device* native_device, DeviceData& device_data) override
   {
      auto& game_device_data = GetGameDeviceData(device_data);

      game_device_data.has_drawn_taa_resolve = false;
      ++game_device_data.frame_index;
   }

   void LoadConfigs() override
   {
      reshade::api::effect_runtime* runtime = nullptr;

      // ReShade.ini is ready here, so load its values.
      reshade::get_config_value(runtime, NAME, "GameSetting01", cb_luma_global_settings.GameSettings.GameSetting01);
      reshade::get_config_value(runtime, NAME, "GameSetting02", cb_luma_global_settings.GameSettings.GameSetting02);
      reshade::get_config_value(runtime, NAME, "CancelTaaResolve", g_cancel_taa_resolve_ini);
   }

   void DrawImGuiSettings(DeviceData& device_data) override
   {
      reshade::api::effect_runtime* runtime = nullptr;
      auto& game_device_data = GetGameDeviceData(device_data);

      ImGui::Text("ETS2 DLAA - v2 stage 1 (resolve isolation)");
      ImGui::Text("TAA resolve 0x805A8981 hits: %u (frame %u)", game_device_data.taa_resolve_hits, game_device_data.frame_index);
      if (game_device_data.frame_index > 0)
         ImGui::Text("hits/frame: %.2f", float(game_device_data.taa_resolve_hits) / float(game_device_data.frame_index));
      ImGui::Text("DLAA: off (stage 2)");
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
      Globals::SetGlobals(cleared_project_name, "ETS2 DLAA v2 stage 1", "", mod_version);

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
