/**
 * Capture Leaflet map instances so we can programmatically clear drawn shapes.
 *
 * dash-leaflet's EditControl.geojson is effectively read-only: setting it to an
 * empty FeatureCollection from Python does NOT remove the visual shapes from the
 * map.  This hook stores every L.Map instance in a global array, and the
 * clientside callback `explore.clearDrawnShapes` iterates those maps to find the
 * draw FeatureGroup and call clearLayers() on it.
 */
L.Map.addInitHook(function () {
    window._leafletMaps = window._leafletMaps || [];
    window._leafletMaps.push(this);
});
