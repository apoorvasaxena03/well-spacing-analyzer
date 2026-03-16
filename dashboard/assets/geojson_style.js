/**
 * Custom Leaflet styling functions for dash-leaflet GeoJSON layers.
 *
 * Both functions read from the `hideout` prop (set by Dash callbacks)
 * to determine per-feature colors, sizes, and opacity.
 *
 * hideout schema:
 *   colorMap:   { featureValue: "#hexcolor", ... }
 *   colorProp:  property name to look up in colorMap (e.g. "bench")
 *   weight:     line thickness (trajectories) or ignored (bottomholes)
 *   opacity:    fill/stroke opacity
 *   radius:     circle marker radius (bottomholes only)
 *   defaultColor: fallback color when feature value not in colorMap
 */
window.dashExtensions = Object.assign({}, window.dashExtensions, {

    // Style function for LineString trajectory features
    trajectoryStyle: function(feature, context) {
        var h = context.hideout || {};
        var colorMap = h.colorMap || {};
        var prop = h.colorProp || "bench";
        var val = (feature.properties || {})[prop];
        var color = colorMap[val] || h.defaultColor || "#3388ff";
        return {
            color: color,
            weight: h.weight || 3,
            opacity: h.opacity || 0.9,
        };
    },

    // pointToLayer for Point bottom-hole features
    bottomholePointToLayer: function(feature, latlng, context) {
        var h = context.hideout || {};
        var colorMap = h.colorMap || {};
        var prop = h.colorProp || "bench";
        var val = (feature.properties || {})[prop];
        var color = colorMap[val] || h.defaultColor || "#e74c3c";
        return L.circleMarker(latlng, {
            radius: h.radius || 4,
            fillColor: color,
            color: "#fff",
            weight: 1,
            fillOpacity: h.opacity || 0.8,
        });
    },
});
