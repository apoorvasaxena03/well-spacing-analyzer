/**
 * Leaflet map hooks for the Well Spacing Analyzer dashboard.
 *
 * 1. Stores every L.Map instance so clientside callbacks can clear drawn shapes.
 * 2. Detects empty-map clicks and programmatically clicks "Clear Selection"
 *    (dash-leaflet's click_lat_lng is unreliable after draw-tool usage).
 *
 * Relies on window._wellFeatureClicked being set by geojson_style.js when a
 * well trajectory or bottomhole is clicked.
 */
L.Map.addInitHook(function () {
    var map = this;

    // ── Store map reference for clientside clearDrawnShapes callback ──
    window._leafletMaps = window._leafletMaps || [];
    window._leafletMaps.push(map);

    // ── Empty-map click → trigger Clear Selection button ──
    map.on('click', function () {
        // Don't clear while a draw tool is active (cursor is crosshair)
        if (map.getContainer().classList.contains('leaflet-crosshair')) return;

        // Wait briefly — feature click handlers fire synchronously before
        // this timeout, so _wellFeatureClicked will be set if a well was hit.
        setTimeout(function () {
            if (window._wellFeatureClicked) {
                window._wellFeatureClicked = false;
                return;  // well was clicked, not empty space
            }
            var btn = document.getElementById('btn-clear-selection');
            if (btn) btn.click();
        }, 100);
    });
});
