
async function send() {
    const input = document.getElementById("input");
    const messages = document.getElementById("messages");

    const question = input.value;
    if (!question) return;

    messages.innerHTML += "<div class='user'>" + question + "</div>";
    input.value = "";

    const response = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question})
    });

    const data = await response.json();
    messages.innerHTML += "<div class='bot'>" + data.answer + "</div>";
}
