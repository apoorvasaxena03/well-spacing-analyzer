/**
 * Throttled scroll-zoom for the Leaflet map.
 *
 * Modern precision mice send many wheel delta events per physical tick,
 * causing 1 scroll = 3-4 zoom levels with Leaflet's native handler.
 *
 * This script finds the actual Leaflet map instance via the
 * leaflet-container class, disables native scroll zoom, and attaches
 * a throttled handler: 1 zoom step per THROTTLE_MS.
 */
(function () {
    var THROTTLE_MS = 400;
    var attached = false;

    function tryAttach() {
        if (attached) return;

        // dash-leaflet renders a div with class "leaflet-container"
        // inside the #main-map wrapper
        var container = document.querySelector("#main-map .leaflet-container");
        if (!container) return;

        // Leaflet stores the map instance on the container element
        // under a key like "_leaflet_id" — we can access it via
        // the undocumented but stable L.DomUtil pattern
        var map = null;
        // Check all properties for a Leaflet map instance
        for (var prop in container) {
            if (prop.indexOf("_leaflet") === 0 && container[prop] &&
                typeof container[prop] === "object" && container[prop].getZoom) {
                map = container[prop];
                break;
            }
        }

        // Alternative: Leaflet stores map id, and we can get it from
        // the internal store
        if (!map && window.L && container._leaflet_id) {
            // Try L.Map instances — not directly accessible, but
            // we can use the eachLayer trick
        }

        if (!map) {
            // Fallback: search for map in the Leaflet internal store
            // Leaflet >=1.0 uses a numeric id on the container
            if (container._leaflet_id !== undefined && window.L) {
                // The map registers event listeners on the container.
                // We can find it by checking stored instances.
            }
        }

        // If we still can't find the map programmatically, use a
        // simpler approach: intercept wheel at the container level
        // and use the +/- zoom buttons programmatically via the
        // Leaflet API on the React component
        if (!map) {
            // Last resort: just throttle the wheel event and let
            // only 1 through per THROTTLE_MS. This works because
            // Leaflet listens on the same container.
            var lastTime = 0;
            container.addEventListener("wheel", function (e) {
                var now = Date.now();
                if (now - lastTime < THROTTLE_MS) {
                    e.stopPropagation();
                    e.preventDefault();
                    return;
                }
                lastTime = now;
                // Let this one event through to Leaflet's native handler
            }, {capture: true, passive: false});

            attached = true;
            console.log("[scroll_zoom] Throttle attached (capture mode, " + THROTTLE_MS + "ms)");
            return;
        }

        // If we found the map instance, disable native and do our own
        if (map.scrollWheelZoom) {
            map.scrollWheelZoom.disable();
        }

        var lastZoomTime = 0;
        container.addEventListener("wheel", function (e) {
            e.preventDefault();
            e.stopPropagation();
            var now = Date.now();
            if (now - lastZoomTime < THROTTLE_MS) return;
            lastZoomTime = now;
            var direction = e.deltaY < 0 ? 1 : -1;
            map.setZoom(map.getZoom() + direction, {animate: true});
        }, {passive: false});

        attached = true;
        console.log("[scroll_zoom] Throttle attached (direct mode, " + THROTTLE_MS + "ms)");
    }

    // Poll until the map container is rendered
    var attempts = 0;
    var poller = setInterval(function () {
        tryAttach();
        attempts++;
        if (attached || attempts > 100) clearInterval(poller);
    }, 500);
})();
