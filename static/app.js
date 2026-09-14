document.addEventListener('DOMContentLoaded', function () {
  const meterFills = document.querySelectorAll('.meter-fill');

  meterFills.forEach(function (fill) {
    const confidence = Number(fill.dataset.confidence || 0);

    if (!Number.isNaN(confidence)) {
      fill.style.width = confidence + '%';
    }
  });
});
