function setUpPresentationMode() {
  if (document.querySelector(".presentation-mode-button")) return;

  var presentationButton = document.createElement("button");
  presentationButton.type = "button";
  presentationButton.className = "presentation-mode-button";
  presentationButton.textContent = "Presentation mode";
  presentationButton.setAttribute("aria-pressed", "false");
  presentationButton.setAttribute(
    "aria-label",
    "Toggle presentation mode"
  );

  presentationButton.addEventListener("click", function () {
    var active = document.body.classList.toggle("presentation-mode");
    presentationButton.textContent = active
      ? "Exit presentation mode"
      : "Presentation mode";
    presentationButton.setAttribute("aria-pressed", String(active));
  });

  document.body.appendChild(presentationButton);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", setUpPresentationMode);
} else {
  setUpPresentationMode();
}
