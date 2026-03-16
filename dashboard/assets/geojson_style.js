/**
 * Custom Leaflet styling functions for dash-leaflet GeoJSON layers.
 * Referenced via the `options` prop in dl.GeoJSON components.
 */
window.dashExtensions = Object.assign({}, window.dashExtensions, {
    // Render Point features as small circle markers instead of default pin icons
    bottomholePointToLayer: function(feature, latlng) {
        return L.circleMarker(latlng, {
            radius: 4,
            fillColor: "#e74c3c",
            color: "#fff",
            weight: 1,
            fillOpacity: 0.8,
        });
    },
});
