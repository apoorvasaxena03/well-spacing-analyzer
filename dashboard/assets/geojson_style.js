window.dashExtensions = window.dashExtensions || {};
window.dashExtensions.default = window.dashExtensions.default || {};

// Flag used by draw_clear.js to distinguish well clicks from empty-map clicks
window._wellFeatureClicked = false;

// Trajectory style — reads colorMap/colorProp/weight/opacity from hideout dict
// set by the update_trajectory_style callback in 05_explore.py.
window.dashExtensions.default.style0 = function(feature, context) {
    var h = context.hideout || {};
    var colorMap = h.colorMap || {};
    var colorProp = h.colorProp || "bench";
    var weight = h.weight || 3;
    var opacity = h.opacity || 0.9;
    var defaultColor = h.defaultColor || "#3388ff";

    var val = (feature.properties || {})[colorProp];
    var color = (val !== undefined && val !== null && val !== "")
        ? (colorMap[String(val)] || defaultColor)
        : defaultColor;

    return {color: color, weight: weight, opacity: opacity};
};

// Bottomhole style — reads colorMap/colorProp/radius/opacity from hideout
window.dashExtensions.default.ptl_colored = function(feature, latlng, context) {
    var h = context.hideout || {};
    var colorMap = h.colorMap || {};
    var colorProp = h.colorProp || "spud_year";
    var size = h.radius || 4;
    var opacity = h.opacity || 0.8;
    var defaultColor = h.defaultColor || "#e74c3c";

    var val = (feature.properties || {})[colorProp];
    var color = (val !== undefined && val !== null && val !== "")
        ? (colorMap[String(val)] || defaultColor)
        : defaultColor;

    return L.circleMarker(latlng, {
        radius: size,
        fillColor: color,
        color: "#333",
        weight: 1,
        fillOpacity: opacity
    });
};

// Bottomhole: solid filled circle markers (legacy — no color-by)
window.dashExtensions.default.ptl0 = function(feature, latlng) {
    return L.circleMarker(latlng, {
        radius: 5,
        fillColor: "#e74c3c",
        color: "#333",
        weight: 1,
        fillOpacity: 0.9
    });
};

// Trajectory tooltip + popup
window.dashExtensions.default.oef0 = function(feature, layer) {
    // Signal that a well feature was clicked (checked by draw_clear.js)
    layer.on('click', function() { window._wellFeatureClicked = true; });

    var p = feature.properties || {};
    // Tooltip on hover
    var tip = (p.well_name || p.uwi || "Well");
    if (p.bench) tip += " | " + p.bench;
    if (p.operator) tip += " | " + p.operator;
    layer.bindTooltip(tip, {sticky: true, direction: "top"});
    // Popup on click
    var rows = [];
    rows.push("<b>" + (p.well_name || "-") + "</b>");
    var fields = [
        ["UWI", p.uwi], ["Operator", p.operator], ["Bench", p.bench],
        ["RSV Cat", p.rsv_cat], ["Spud", p.spud_date ? String(p.spud_date).substring(0,10) : null],
        ["First Prod", p.first_prod_date ? String(p.first_prod_date).substring(0,10) : null],
        ["Lateral (ft)", p.lateral_length_ft ? Math.round(p.lateral_length_ft).toLocaleString() : null],
    ];
    for (var i = 0; i < fields.length; i++) {
        if (fields[i][1] && fields[i][1] !== "" && fields[i][1] !== "nan")
            rows.push("<tr><td style='padding:1px 6px;color:#666'>" + fields[i][0] + "</td><td style='padding:1px 6px'>" + fields[i][1] + "</td></tr>");
    }
    var title = rows.shift();
    layer.bindPopup("<div style='font-size:12px'>" + title + "<table style='margin-top:4px'>" + rows.join("") + "</table></div>", {maxWidth: 300});
};

// Bottomhole tooltip + popup (same as trajectory)
window.dashExtensions.default.oef1 = function(feature, layer) {
    window.dashExtensions.default.oef0(feature, layer);
};
