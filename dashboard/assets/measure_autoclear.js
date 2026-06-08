/**
 * Auto-clear completed measurement lines from leaflet-measure.
 *
 * The MeasureControl plugin leaves result overlays on the map after
 * each measurement. This script observes the DOM for completed
 * measurement elements and removes them after a brief display period
 * (3 seconds — enough to read the result).
 *
 * Works with dash-leaflet 1.1.3 / leaflet-measure.
 */
(function () {
    "use strict";

    // Time (ms) to keep the measurement result visible before auto-clearing
    var DISPLAY_MS = 4000;

    // MutationObserver watches for new measurement result popups
    var observer = new MutationObserver(function (mutations) {
        mutations.forEach(function (mutation) {
            mutation.addedNodes.forEach(function (node) {
                if (!(node instanceof HTMLElement)) return;

                // leaflet-measure adds elements with class "leaflet-measure-resultpopup"
                // or "js-results" inside the measure control
                var results = node.querySelectorAll
                    ? node.querySelectorAll(".leaflet-measure-resultpopup")
                    : [];

                // Also check if the node itself is a result popup
                if (node.classList && node.classList.contains("leaflet-measure-resultpopup")) {
                    scheduleRemoval(node);
                }

                results.forEach(function (el) {
                    scheduleRemoval(el);
                });
            });
        });
    });

    function scheduleRemoval(el) {
        // Don't double-schedule
        if (el.dataset.autoRemoveScheduled) return;
        el.dataset.autoRemoveScheduled = "true";

        setTimeout(function () {
            // Remove the result popup and any associated path layers
            if (el && el.parentNode) {
                el.parentNode.removeChild(el);
            }

            // Also clean up any completed measure paths on the map
            // These have the completedColor style (#00C853)
            document.querySelectorAll("path[stroke='#00C853']").forEach(function (path) {
                if (path.parentNode) {
                    path.parentNode.removeChild(path);
                }
            });
        }, DISPLAY_MS);
    }

    // Start observing once the DOM is ready
    function init() {
        var mapContainer = document.querySelector(".leaflet-container");
        if (mapContainer) {
            observer.observe(mapContainer, { childList: true, subtree: true });
        } else {
            // Map not ready yet — retry
            setTimeout(init, 1000);
        }
    }

    // Wait for page load
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        setTimeout(init, 1000);
    }
})();
