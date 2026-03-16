window.dashExtensions = window.dashExtensions || {};
window.dashExtensions.default = window.dashExtensions.default || {};

// Bottomhole: solid filled circle markers
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
