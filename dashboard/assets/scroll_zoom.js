/**
 * Throttled scroll-zoom for the Leaflet map.
 *
 * Uses a MutationObserver to detect when .leaflet-container appears
 * (which can happen any time the user navigates to the Explore page),
 * then attaches a capture-phase wheel listener that blocks rapid-fire
 * scroll events — only 1 wheel event per THROTTLE_MS passes through.
 */
(function () {
    var THROTTLE_MS = 1000;
    var attachedContainers = new WeakSet();

    function throttleContainer(container) {
        if (attachedContainers.has(container)) return;
        attachedContainers.add(container);

        var lastTime = 0;
        container.addEventListener("wheel", function (e) {
            var now = Date.now();
            if (now - lastTime < THROTTLE_MS) {
                e.stopImmediatePropagation();
                e.preventDefault();
                return false;
            }
            lastTime = now;
        }, {capture: true, passive: false});

        console.log("[scroll_zoom] Throttle attached (" + THROTTLE_MS + "ms)");
    }

    // Watch the entire document for leaflet-container elements
    var observer = new MutationObserver(function () {
        var containers = document.querySelectorAll(".leaflet-container");
        for (var i = 0; i < containers.length; i++) {
            throttleContainer(containers[i]);
        }
    });
    observer.observe(document.body, {childList: true, subtree: true});

    // Also check immediately
    var containers = document.querySelectorAll(".leaflet-container");
    for (var i = 0; i < containers.length; i++) {
        throttleContainer(containers[i]);
    }
})();
