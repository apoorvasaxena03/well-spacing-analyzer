/**
 * Custom Leaflet styling functions for dash-leaflet GeoJSON layers.
 *
 * Uses window.dash_props_handler namespace which dash-leaflet resolves
 * when a string like "dash_props_handler.trajectoryStyle" is passed
 * as a style/pointToLayer/onEachFeature prop.
 *
 * hideout schema (passed from Dash callback):
 *   colorMap:     { featureValue: "#hexcolor", ... }
 *   colorProp:    property name to look up in colorMap (e.g. "bench")
 *   weight:       line thickness (trajectories)
 *   opacity:      fill/stroke opacity
 *   radius:       circle marker radius (bottomholes only)
 *   defaultColor: fallback color when feature value not in colorMap
 */

// Ensure namespace exists
window.dashExtensions = window.dashExtensions || {};
window.dash_props_handler = window.dash_props_handler || {};

// ── Style function for LineString trajectory features ──
function trajectoryStyleFn(feature, context) {
    var h = (context && context.hideout) || {};
    var colorMap = h.colorMap || {};
    var prop = h.colorProp || "bench";
    var val = (feature.properties || {})[prop];
    var color = (val && colorMap[val]) || h.defaultColor || "#3388ff";
    return {
        color: color,
        weight: h.weight || 3,
        opacity: h.opacity || 0.9,
    };
}

// ── pointToLayer for Point bottom-hole features ──
function bottomholePtlFn(feature, latlng, context) {
    var h = (context && context.hideout) || {};
    var colorMap = h.colorMap || {};
    var prop = h.colorProp || "bench";
    var val = (feature.properties || {})[prop];
    var color = (val && colorMap[val]) || h.defaultColor || "#e74c3c";
    return L.circleMarker(latlng, {
        radius: h.radius || 4,
        fillColor: color,
        color: "#fff",
        weight: 1,
        fillOpacity: h.opacity || 0.8,
    });
}

// ── onEachFeature: tooltip + popup for trajectories ──
function trajectoryOnEachFn(feature, layer) {
    var p = feature.properties || {};
    var tip = (p.well_name || p.uwi || "Unknown");
    if (p.bench) tip += " | " + p.bench;
    if (p.operator) tip += " | " + p.operator;
    layer.bindTooltip(tip, {sticky: true, direction: "top", className: "well-tooltip"});

    var rows = [];
    rows.push("<b>" + (p.well_name || "—") + "</b>");
    if (p.uwi) rows.push("<tr><td style='padding:1px 6px;color:#666'>UWI</td><td style='padding:1px 6px'>" + p.uwi + "</td></tr>");
    if (p.operator) rows.push("<tr><td style='padding:1px 6px;color:#666'>Operator</td><td style='padding:1px 6px'>" + p.operator + "</td></tr>");
    if (p.bench) rows.push("<tr><td style='padding:1px 6px;color:#666'>Bench</td><td style='padding:1px 6px'>" + p.bench + "</td></tr>");
    if (p.rsv_cat) rows.push("<tr><td style='padding:1px 6px;color:#666'>RSV Cat</td><td style='padding:1px 6px'>" + p.rsv_cat + "</td></tr>");
    if (p.spud_date && p.spud_date !== "") rows.push("<tr><td style='padding:1px 6px;color:#666'>Spud</td><td style='padding:1px 6px'>" + String(p.spud_date).substring(0,10) + "</td></tr>");
    if (p.first_prod_date && p.first_prod_date !== "") rows.push("<tr><td style='padding:1px 6px;color:#666'>First Prod</td><td style='padding:1px 6px'>" + String(p.first_prod_date).substring(0,10) + "</td></tr>");
    if (p.lateral_length_ft) rows.push("<tr><td style='padding:1px 6px;color:#666'>Lateral (ft)</td><td style='padding:1px 6px'>" + Math.round(p.lateral_length_ft).toLocaleString() + "</td></tr>");

    var title = rows.shift();
    var html = "<div style='font-size:12px'>" + title +
               "<table style='margin-top:4px;border-collapse:collapse'>" + rows.join("") + "</table></div>";
    layer.bindPopup(html, {maxWidth: 300});
}

// ── onEachFeature: tooltip + popup for bottomholes ──
function bottomholeOnEachFn(feature, layer) {
    var p = feature.properties || {};
    var tip = (p.well_name || p.uwi || "Unknown");
    if (p.bench) tip += " | " + p.bench;
    layer.bindTooltip(tip, {sticky: true, direction: "top", className: "well-tooltip"});

    var rows = [];
    rows.push("<b>" + (p.well_name || "—") + "</b>");
    if (p.uwi) rows.push("<tr><td style='padding:1px 6px;color:#666'>UWI</td><td style='padding:1px 6px'>" + p.uwi + "</td></tr>");
    if (p.operator) rows.push("<tr><td style='padding:1px 6px;color:#666'>Operator</td><td style='padding:1px 6px'>" + p.operator + "</td></tr>");
    if (p.bench) rows.push("<tr><td style='padding:1px 6px;color:#666'>Bench</td><td style='padding:1px 6px'>" + p.bench + "</td></tr>");
    if (p.rsv_cat) rows.push("<tr><td style='padding:1px 6px;color:#666'>RSV Cat</td><td style='padding:1px 6px'>" + p.rsv_cat + "</td></tr>");

    var title = rows.shift();
    var html = "<div style='font-size:12px'>" + title +
               "<table style='margin-top:4px;border-collapse:collapse'>" + rows.join("") + "</table></div>";
    layer.bindPopup(html, {maxWidth: 300});
}

// Register in both namespaces (dashExtensions for dash-leaflet, dash_props_handler as fallback)
window.dashExtensions.trajectoryStyle = trajectoryStyleFn;
window.dashExtensions.bottomholePointToLayer = bottomholePtlFn;
window.dashExtensions.trajectoryOnEach = trajectoryOnEachFn;
window.dashExtensions.bottomholeOnEach = bottomholeOnEachFn;

window.dash_props_handler.trajectoryStyle = trajectoryStyleFn;
window.dash_props_handler.bottomholePointToLayer = bottomholePtlFn;
window.dash_props_handler.trajectoryOnEach = trajectoryOnEachFn;
window.dash_props_handler.bottomholeOnEach = bottomholeOnEachFn;
