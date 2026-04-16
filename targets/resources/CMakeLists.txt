set(LOC_LANG_OUT ${CMAKE_CURRENT_BINARY_DIR}/languages.loc)
set(LOC_STRINGS_H ${CMAKE_CURRENT_BINARY_DIR}/strings.h)

add_custom_command(
    OUTPUT ${LOC_LANG_OUT} ${LOC_STRINGS_H}
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/pack_loc.py
            ${CMAKE_CURRENT_SOURCE_DIR}/Windows64Media/loc
            ${LOC_LANG_OUT}
            ${LOC_STRINGS_H}
    DEPENDS ${CMAKE_SOURCE_DIR}/scripts/pack_loc.py
    COMMENT "Packing localisation"
    VERBATIM
)

add_custom_target(plce_resources_localisation DEPENDS ${LOC_LANG_OUT} ${LOC_STRINGS_H})

set(COLOUR_TABLE_OUT ${CMAKE_CURRENT_BINARY_DIR}/HTMLColours.col)
add_custom_command(
    OUTPUT ${COLOUR_TABLE_OUT}
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/build_col.py
            ${CMAKE_CURRENT_SOURCE_DIR}/Common/Media/HTMLColours.xml
            ${COLOUR_TABLE_OUT}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Common/Media/HTMLColours.xml
            ${CMAKE_SOURCE_DIR}/scripts/build_col.py
    COMMENT "Building colour table"
    VERBATIM
)

add_custom_target(plce_resources_colour_table DEPENDS ${COLOUR_TABLE_OUT})

# Archive sources mirror meson.build's lists
set(ARCHIVE_SOURCES_COMMON
    ${LOC_LANG_OUT}
    ${COLOUR_TABLE_OUT}
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/Media/splashes.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/Media/Graphics/SaveChest.png
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/Media/Graphics/MinecraftIcon.png
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/Media/Graphics/TexturePackIcon.png
)

# Helper: build a list of Common/Media/<file> paths
set(_movies_1080_rel
    skinHDGraphics.swf skinHDGraphicsHud.swf skinHDGraphicsLabels.swf skinHDGraphicsInGame.swf
    skinHD.swf skinHDHud.swf skinHDLabels.swf skinHDInGame.swf
    AnvilMenu1080.swf BeaconMenu1080.swf BrewingStandMenu1080.swf ChestMenu1080.swf
    ChestLargeMenu1080.swf ComponentLogo1080.swf ComponentLogoSplit1080.swf Controls1080.swf
    ControlsRemotePlay1080.swf CreateWorldMenu1080.swf CreativeMenu1080.swf Credits1080.swf
    Crafting2x2Menu1080.swf Crafting3x3Menu1080.swf DeathMenu1080.swf
    DebugCreateSchematic1080.swf DebugMenu1080.swf DebugOptionsMenu1080.swf DebugSetCamera1080.swf
    DebugUIConsoleComponent1080.swf DebugUIMarketingGuide1080.swf DLCMainMenu1080.swf
    DispenserMenu1080.swf EnchantingMenu1080.swf EndPoem1080.swf EULA1080.swf
    FireworksMenu1080.swf FullscreenProgress1080.swf FurnaceMenu1080.swf HelpAndOptionsMenu1080.swf
    HopperMenu1080.swf HorseInventoryMenu1080.swf HowToPlay1080.swf HowToPlayMenu1080.swf
    HUD1080.swf InGameHostOptions1080.swf InGameInfoMenu1080.swf InGamePlayerOptions1080.swf
    InGameTeleportMenu1080.swf Intro1080.swf InventoryMenu1080.swf JoinMenu1080.swf
    LanguagesMenu1080.swf LanguagesMenuSplit1080.swf LoadOrJoinMenu1080.swf
    LaunchMoreOptionsMenu1080.swf LeaderboardMenu1080.swf LoadMenu1080.swf MainMenu1080.swf
    MenuBackground1080.swf MessageBox1080.swf NewUpdateMessage1080.swf Panorama1080.swf
    PauseMenu1080.swf PressStartToPlay1080.swf QuadrantSignin1080.swf ReinstallMenu1080.swf
    SaveMenu1080.swf SaveMessage1080.swf SettingsMenu1080.swf SettingsAudioMenu1080.swf
    SettingsControlMenu1080.swf SettingsGraphicsMenu1080.swf SettingsOptionsMenu1080.swf
    SettingsUIMenu1080.swf SignEntryMenu1080.swf SkinSelectMenu1080.swf Timer1080.swf
    ToolTips1080.swf TradingMenu1080.swf TutorialPopup1080.swf
    AnvilMenuSplit1080.swf BeaconMenuSplit1080.swf BrewingStandMenuSplit1080.swf
    ChestMenuSplit1080.swf ChestLargeMenuSplit1080.swf ControlsSplit1080.swf
    Crafting2x2MenuSplit1080.swf Crafting3x3MenuSplit1080.swf CreativeMenuSplit1080.swf
    DeathMenuSplit1080.swf DispenserMenuSplit1080.swf EnchantingMenuSplit1080.swf
    FireworksMenuSplit1080.swf FurnaceMenuSplit1080.swf FullscreenProgressSplit1080.swf
    HelpAndOptionsMenuSplit1080.swf HopperMenuSplit1080.swf HorseInventoryMenuSplit1080.swf
    HowToPlaySplit1080.swf HowToPlayMenuSplit1080.swf HUDSplit1080.swf
    InGameHostOptionsSplit1080.swf InGameInfoMenuSplit1080.swf InGameTeleportMenuSplit1080.swf
    InGamePlayerOptionsSplit1080.swf InventoryMenuSplit1080.swf MessageBoxSplit1080.swf
    PanoramaSplit1080.swf PauseMenuSplit1080.swf ReinstallMenuSplit1080.swf
    SettingsAudioMenuSplit1080.swf SettingsControlMenuSplit1080.swf
    SettingsGraphicsMenuSplit1080.swf SettingsMenuSplit1080.swf SettingsOptionsMenuSplit1080.swf
    SettingsUIMenuSplit1080.swf SignEntryMenuSplit1080.swf SkinSelectMenuSplit1080.swf
    TimerSplit1080.swf ToolTipsSplit1080.swf TradingMenuSplit1080.swf TutorialPopupSplit1080.swf
    Keyboard1080.swf KeyboardSplit1080.swf
)

set(_movies_720_rel
    skinGraphics.swf skinGraphicsHud.swf skinGraphicsLabels.swf skinGraphicsInGame.swf
    skin.swf skinHud.swf skinLabels.swf skinInGame.swf
    AnvilMenu720.swf BrewingStandMenu720.swf ChestMenu720.swf ChestLargeMenu720.swf
    ComponentLogo720.swf Controls720.swf CreateWorldMenu720.swf CreativeMenu720.swf
    Credits720.swf Crafting2x2Menu720.swf Crafting3x3Menu720.swf DeathMenu720.swf
    DebugCreateSchematic720.swf DebugMenu720.swf DebugOptionsMenu720.swf DebugSetCamera720.swf
    DebugUIConsoleComponent720.swf DebugUIMarketingGuide720.swf DLCMainMenu720.swf
    DispenserMenu720.swf EnchantingMenu720.swf EndPoem720.swf EULA720.swf
    FullscreenProgress720.swf FurnaceMenu720.swf HelpAndOptionsMenu720.swf HowToPlay720.swf
    HowToPlayMenu720.swf HUD720.swf InGameHostOptions720.swf InGameInfoMenu720.swf
    InGamePlayerOptions720.swf InGameTeleportMenu720.swf Intro720.swf InventoryMenu720.swf
    JoinMenu720.swf LoadOrJoinMenu720.swf LaunchMoreOptionsMenu720.swf LeaderboardMenu720.swf
    LoadMenu720.swf MainMenu720.swf MenuBackground720.swf MessageBox720.swf
    NewUpdateMessage720.swf Panorama720.swf PauseMenu720.swf PressStartToPlay720.swf
    QuadrantSignin720.swf ReinstallMenu720.swf SaveMessage720.swf SettingsMenu720.swf
    SettingsAudioMenu720.swf SettingsControlMenu720.swf SettingsGraphicsMenu720.swf
    SettingsOptionsMenu720.swf SettingsUIMenu720.swf SignEntryMenu720.swf SkinSelectMenu720.swf
    Timer720.swf ToolTips720.swf TradingMenu720.swf TutorialPopup720.swf
    AnvilMenuSplit720.swf BrewingStandMenuSplit720.swf ChestMenuSplit720.swf
    ChestLargeMenuSplit720.swf ControlsSplit720.swf ComponentLogoSplit720.swf
    Crafting2x2MenuSplit720.swf Crafting3x3MenuSplit720.swf CreativeMenuSplit720.swf
    DeathMenuSplit720.swf DispenserMenuSplit720.swf EnchantingMenuSplit720.swf
    FurnaceMenuSplit720.swf FullscreenProgressSplit720.swf GamertagSplit720.swf
    HelpAndOptionsMenuSplit720.swf HowToPlaySplit720.swf HowToPlayMenuSplit720.swf
    HUDSplit720.swf InGameHostOptionsSplit720.swf InGameInfoMenuSplit720.swf
    InGamePlayerOptionsSplit720.swf InventoryMenuSplit720.swf MessageBoxSplit720.swf
    PanoramaSplit720.swf PauseMenuSplit720.swf ReinstallMenuSplit720.swf
    SettingsAudioMenuSplit720.swf SettingsControlMenuSplit720.swf
    SettingsGraphicsMenuSplit720.swf SettingsMenuSplit720.swf SettingsOptionsMenuSplit720.swf
    SettingsUIMenuSplit720.swf SignEntryMenuSplit720.swf InGameTeleportMenuSplit720.swf
    ToolTipsSplit720.swf TradingMenuSplit720.swf TrialExitUpsell720.swf
    TutorialPopupSplit720.swf SkinSelectMenuSplit720.swf
)

set(_windows64_media_rel
    DLCOffersMenu1080.swf DLCOffersMenu480.swf DLCOffersMenu720.swf DLCOffersMenuVita.swf
    Tutorial.pck skinHDWin.swf skinWin.swf
)

set(_media_archive_inputs ${ARCHIVE_SOURCES_COMMON})
foreach(f IN LISTS _movies_1080_rel _movies_720_rel)
    list(APPEND _media_archive_inputs ${CMAKE_CURRENT_SOURCE_DIR}/Common/Media/${f})
endforeach()
foreach(f IN LISTS _windows64_media_rel)
    list(APPEND _media_archive_inputs ${CMAKE_CURRENT_SOURCE_DIR}/Windows64Media/Media/${f})
endforeach()

set(MEDIA_ARCHIVE_OUT ${CMAKE_CURRENT_BINARY_DIR}/MediaWindows64.arc
    CACHE INTERNAL "Path to MediaWindows64.arc, consumed by targets/app")
add_custom_command(
    OUTPUT ${MEDIA_ARCHIVE_OUT}
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/pack_arc.py
            ${MEDIA_ARCHIVE_OUT}
            ${_media_archive_inputs}
    DEPENDS ${_media_archive_inputs}
            ${CMAKE_SOURCE_DIR}/scripts/pack_arc.py
    COMMENT "Packing media archive"
    VERBATIM
    COMMAND_EXPAND_LISTS
)

add_custom_target(plce_media_archive ALL DEPENDS ${MEDIA_ARCHIVE_OUT})

# Exposes the generated strings.h to downstream targets.
add_library(plce_assets_localisation INTERFACE)
add_dependencies(plce_assets_localisation plce_resources_localisation)
target_include_directories(plce_assets_localisation INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

install(DIRECTORY Common/res     DESTINATION Common)
install(DIRECTORY Common/Trial   DESTINATION Common)
install(DIRECTORY Common/music   DESTINATION .)
install(DIRECTORY DurangoMedia/Sound DESTINATION .)
install(FILES ${MEDIA_ARCHIVE_OUT} DESTINATION Common/Media)
