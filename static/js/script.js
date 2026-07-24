// Wait until the page is fully loaded
document.addEventListener("DOMContentLoaded", function () {

    const form = document.querySelector("form");
    const button = document.querySelector("button[type='submit']");

    form.addEventListener("submit", function () {

        button.disabled = true;
        button.innerText = "Predicting...";

    });

});