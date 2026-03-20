/**
 * Custom scroll-zoom handler for the Leaflet map.
 *
 * Leaflet's built-in scrollWheelZoom fires on every wheel delta event,
 * and modern precision mice/trackpads send many deltas per physical
 * scroll tick, causing 1 scroll = 3-4 zoom levels.
 *
 * This script disables native scroll zoom and replaces it with a
 * throttled handler: at most 1 zoom step per 400 ms, accumulating
 * scroll direction but ignoring magnitude.
 */
document.addEventListener("DOMContentLoaded", function () {
    var THROTTLE_MS = 400;

    function attach(mapEl) {
        // dash-leaflet stores the Leaflet map instance on the DOM element
        var poll = setInterval(function () {
            // Access the Leaflet map via the _leaflet_map property or
            // iterate window._leaflet_id. dash-leaflet >=1.0 exposes it
            // on the container's _leaflet_map or via el._leaflet_id.
            var map = null;
            if (mapEl._leaflet_map) {
                map = mapEl._leaflet_map;
            } else if (mapEl._leaflet_id !== undefined) {
                // Leaflet stores map instances keyed by id
                for (var key in window) {
                    if (window[key] && window[key]._leaflet_id === mapEl._leaflet_id) {
                        map = window[key];
                        break;
                    }
                }
            }
            if (!map || !map.getZoom) return;
            clearInterval(poll);

            // Ensure native scroll zoom is off
            if (map.scrollWheelZoom && map.scrollWheelZoom.enabled()) {
                map.scrollWheelZoom.disable();
            }

            var lastZoomTime = 0;
            mapEl.addEventListener("wheel", function (e) {
                e.preventDefault();
                e.stopPropagation();
                var now = Date.now();
                if (now - lastZoomTime < THROTTLE_MS) return;
                lastZoomTime = now;
                var direction = e.deltaY < 0 ? 1 : -1;  // up = zoom in
                map.setZoom(map.getZoom() + direction, {animate: true});
            }, {passive: false});
        }, 300);
    }

    // Wait for the map container to appear in the DOM
    var observer = new MutationObserver(function () {
        var mapEl = document.getElementById("main-map");
        if (mapEl) {
            observer.disconnect();
            attach(mapEl);
        }
    });
    observer.observe(document.body, {childList: true, subtree: true});

    // Also try immediately in case it's already rendered
    var mapEl = document.getElementById("main-map");
    if (mapEl) attach(mapEl);
});
