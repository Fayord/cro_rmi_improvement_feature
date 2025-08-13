(function () {
	function getCyInstance(el) {
		try {
			return (el && (
				(el._cyreg && el._cyreg.cy) ||
				el.cy ||
				(el.__cy && el.__cy) ||
				el._cy
			)) || null;
		} catch (e) {
			return null;
		}
	}

	function attachFreeHandler() {
		var container = document.getElementById('my-cytoscape');
		if (!container) {
			setTimeout(attachFreeHandler, 300);
			return;
		}

		var cy = getCyInstance(container);
		if (!cy) {
			setTimeout(attachFreeHandler, 300);
			return;
		}

		if (cy.__freeToTapHooked) return;
		cy.__freeToTapHooked = true;

		cy.on('free', 'node', function (evt) {
			try {
				// Trigger a tap on the node to notify Dash instantly
				evt.target.trigger('tap');
			} catch (e) {
				// no-op
			}
		});
	}

	if (document.readyState === 'loading') {
		document.addEventListener('DOMContentLoaded', attachFreeHandler);
	} else {
		attachFreeHandler();
	}
})();

