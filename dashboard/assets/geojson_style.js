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

    // onEachFeature: hover tooltip + click popup for trajectories
    trajectoryOnEach: function(feature, layer) {
        var p = feature.properties || {};
        // Tooltip (on hover) — compact one-liner
        var tip = (p.well_name || p.uwi || "Unknown");
        if (p.bench) tip += " | " + p.bench;
        if (p.operator) tip += " | " + p.operator;
        layer.bindTooltip(tip, {sticky: true, direction: "top", className: "well-tooltip"});

        // Popup (on click) — detailed well info card
        var rows = [];
        rows.push("<b>" + (p.well_name || "—") + "</b>");
        if (p.uwi) rows.push("<tr><td style='padding:1px 6px;color:#666'>UWI</td><td style='padding:1px 6px'>" + p.uwi + "</td></tr>");
        if (p.operator) rows.push("<tr><td style='padding:1px 6px;color:#666'>Operator</td><td style='padding:1px 6px'>" + p.operator + "</td></tr>");
        if (p.bench) rows.push("<tr><td style='padding:1px 6px;color:#666'>Bench</td><td style='padding:1px 6px'>" + p.bench + "</td></tr>");
        if (p.rsv_cat) rows.push("<tr><td style='padding:1px 6px;color:#666'>RSV Cat</td><td style='padding:1px 6px'>" + p.rsv_cat + "</td></tr>");
        if (p.spud_date) rows.push("<tr><td style='padding:1px 6px;color:#666'>Spud</td><td style='padding:1px 6px'>" + p.spud_date.substring(0,10) + "</td></tr>");
        if (p.first_prod_date) rows.push("<tr><td style='padding:1px 6px;color:#666'>First Prod</td><td style='padding:1px 6px'>" + p.first_prod_date.substring(0,10) + "</td></tr>");
        if (p.lateral_length_ft) rows.push("<tr><td style='padding:1px 6px;color:#666'>Lateral (ft)</td><td style='padding:1px 6px'>" + Math.round(p.lateral_length_ft).toLocaleString() + "</td></tr>");
        if (p.hole_direction) rows.push("<tr><td style='padding:1px 6px;color:#666'>Direction</td><td style='padding:1px 6px'>" + p.hole_direction + "</td></tr>");

        var title = rows.shift();
        var html = "<div style='font-size:12px'>" + title +
                   "<table style='margin-top:4px;border-collapse:collapse'>" + rows.join("") + "</table></div>";
        layer.bindPopup(html, {maxWidth: 300});
    },

    // onEachFeature: hover tooltip + click popup for bottomholes
    bottomholeOnEach: function(feature, layer) {
        var p = feature.properties || {};
        var tip = (p.well_name || p.uwi || "Unknown");
        if (p.bench) tip += " | " + p.bench;
        layer.bindTooltip(tip, {sticky: true, direction: "top", className: "well-tooltip"});

        // Same popup as trajectory
        var rows = [];
        rows.push("<b>" + (p.well_name || "—") + "</b>");
        if (p.uwi) rows.push("<tr><td style='padding:1px 6px;color:#666'>UWI</td><td style='padding:1px 6px'>" + p.uwi + "</td></tr>");
        if (p.operator) rows.push("<tr><td style='padding:1px 6px;color:#666'>Operator</td><td style='padding:1px 6px'>" + p.operator + "</td></tr>");
        if (p.bench) rows.push("<tr><td style='padding:1px 6px;color:#666'>Bench</td><td style='padding:1px 6px'>" + p.bench + "</td></tr>");
        if (p.rsv_cat) rows.push("<tr><td style='padding:1px 6px;color:#666'>RSV Cat</td><td style='padding:1px 6px'>" + p.rsv_cat + "</td></tr>");
        if (p.spud_date) rows.push("<tr><td style='padding:1px 6px;color:#666'>Spud</td><td style='padding:1px 6px'>" + p.spud_date.substring(0,10) + "</td></tr>");
        if (p.first_prod_date) rows.push("<tr><td style='padding:1px 6px;color:#666'>First Prod</td><td style='padding:1px 6px'>" + p.first_prod_date.substring(0,10) + "</td></tr>");
        if (p.lateral_length_ft) rows.push("<tr><td style='padding:1px 6px;color:#666'>Lateral (ft)</td><td style='padding:1px 6px'>" + Math.round(p.lateral_length_ft).toLocaleString() + "</td></tr>");

        var title = rows.shift();
        var html = "<div style='font-size:12px'>" + title +
                   "<table style='margin-top:4px;border-collapse:collapse'>" + rows.join("") + "</table></div>";
        layer.bindPopup(html, {maxWidth: 300});
    },
});
