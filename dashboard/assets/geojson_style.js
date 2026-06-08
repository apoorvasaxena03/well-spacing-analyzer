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

// User-configurable tooltip fields — updated from Python via clientside callback
window._tooltipFields = ["well_name", "uwi", "bench", "operator", "role"];

// Helper: format a property key as a readable label
function _formatLabel(key) {
    return key.replace(/_/g, " ").replace(/\b\w/g, function(c) { return c.toUpperCase(); });
}

// Helper: format a property value for display
function _formatValue(val) {
    if (val === null || val === undefined || val === "" || val === "nan" || val === "NaT" || val === "None") return null;
    var s = String(val);
    // Truncate ISO timestamps to date only
    if (s.length >= 10 && s[4] === "-" && s[7] === "-") return s.substring(0, 10);
    // Format numbers with commas — but skip long numeric strings like UWIs (14+ digits)
    if (s.length < 12) {
        var n = Number(val);
        if (!isNaN(n) && s === String(n)) return n.toLocaleString(undefined, {maximumFractionDigits: 1});
    }
    return s;
}

// Build tooltip text from current fields selection
function _buildTooltipText(p) {
    var fields = window._tooltipFields || ["well_name", "uwi", "bench"];
    var parts = [];
    for (var i = 0; i < fields.length; i++) {
        var v = _formatValue(p[fields[i]]);
        if (v) parts.push(v);
    }
    return parts.join(" | ") || (p.uwi || "Well");
}

// Build popup HTML from current fields selection
function _buildPopupHtml(p) {
    var fields = window._tooltipFields || ["well_name", "uwi", "bench"];
    var title = "<b>" + (p.well_name || p.uwi || "Well") + "</b>";
    var rows = [];
    for (var j = 0; j < fields.length; j++) {
        var label = _formatLabel(fields[j]);
        var val = _formatValue(p[fields[j]]);
        if (val) {
            rows.push(
                "<tr><td style='padding:1px 6px;color:#666'>" + label +
                "</td><td style='padding:1px 6px'>" + val + "</td></tr>"
            );
        }
    }
    return "<div style='font-size:12px'>" + title +
        "<table style='margin-top:4px'>" + rows.join("") + "</table></div>";
}

// Trajectory tooltip + popup (dynamically reads window._tooltipFields on each hover/click)
window.dashExtensions.default.oef0 = function(feature, layer) {
    // Signal that a well feature was clicked (checked by draw_clear.js)
    layer.on('click', function() { window._wellFeatureClicked = true; });

    var p = feature.properties || {};

    // Bind tooltip with dynamic content — re-reads _tooltipFields each time
    layer.bindTooltip("", {sticky: true, direction: "top"});
    layer.on('mouseover', function() {
        layer.setTooltipContent(_buildTooltipText(p));
    });

    // Bind popup with dynamic content — re-reads _tooltipFields on each click
    layer.bindPopup("", {maxWidth: 350});
    layer.on('click', function() {
        layer.setPopupContent(_buildPopupHtml(p));
    });
};

// Bottomhole tooltip + popup (same as trajectory)
window.dashExtensions.default.oef1 = function(feature, layer) {
    window.dashExtensions.default.oef0(feature, layer);
};

// ── Neighborhood edge lines ──────────────────────────────────────────────
// Color by alignment type
var _EDGE_COLORS = {
    "parallel_like":  "#2196F3",  // blue
    "perpendicular":  "#4CAF50",  // green
    "oblique":        "#FF5722",  // deep orange
    "unknown":        "#9E9E9E",  // grey
};

// Edge line style — color by alignment, dashed
window.dashExtensions.default.edgeStyle = function(feature, context) {
    var p = feature.properties || {};
    var align = p.alignment || "unknown";
    var color = _EDGE_COLORS[align] || _EDGE_COLORS["unknown"];
    // Thicker lines for closer wells
    var dist = p.h_dist || 1320;
    var weight = dist < 660 ? 3 : (dist < 1000 ? 2 : 1.5);
    return {
        color: color,
        weight: weight,
        opacity: 0.7,
        dashArray: p.same_bench ? "8 4" : "4 4 1 4",  // dashed for same bench, dot-dash for cross-bench
    };
};

// Edge tooltip — shows distance, alignment, well pair
window.dashExtensions.default.edgeTooltip = function(feature, layer) {
    var p = feature.properties || {};
    var align = (p.alignment || "unknown").replace("_", " ");
    var tip = "H: " + (p.h_dist || 0).toLocaleString() + " ft";
    if (p.v_dist) tip += " | V: " + p.v_dist.toLocaleString() + " ft";
    tip += " | " + align;
    if (!p.same_bench) tip += " (cross-bench)";
    layer.bindTooltip(tip, {sticky: true, direction: "top"});

    // Popup with full details
    var rows = [
        "<b>Neighbor Edge</b>",
        "<tr><td style='padding:1px 6px;color:#666'>Well I</td><td style='padding:1px 6px'>" + p.well_i + "</td></tr>",
        "<tr><td style='padding:1px 6px;color:#666'>Well K</td><td style='padding:1px 6px'>" + p.well_k + "</td></tr>",
        "<tr><td style='padding:1px 6px;color:#666'>H Dist</td><td style='padding:1px 6px'>" + (p.h_dist || 0).toLocaleString() + " ft</td></tr>",
        "<tr><td style='padding:1px 6px;color:#666'>V Dist</td><td style='padding:1px 6px'>" + (p.v_dist || 0).toLocaleString() + " ft</td></tr>",
        "<tr><td style='padding:1px 6px;color:#666'>Alignment</td><td style='padding:1px 6px'>" + align + "</td></tr>",
        "<tr><td style='padding:1px 6px;color:#666'>Bench I</td><td style='padding:1px 6px'>" + (p.bench_i || "") + "</td></tr>",
        "<tr><td style='padding:1px 6px;color:#666'>Bench K</td><td style='padding:1px 6px'>" + (p.bench_k || "") + "</td></tr>",
    ];
    var title = rows.shift();
    layer.bindPopup(
        "<div style='font-size:12px'>" + title + "<table style='margin-top:4px'>" + rows.join("") + "</table></div>",
        {maxWidth: 300}
    );
};
