// Live updates on a post page, the only script on the site. The server sends each new comment already rendered, with
// where it goes: the end of its thread's replies, or the end of the threads for a top-level comment. The browser's
// EventSource reconnects by itself, and resumes after the last comment it got. Without it, a reload shows the same.
(() => {
  const section = document.getElementById("comments");
  if (!section || !window.EventSource) return;
  const stream = new EventSource(section.dataset.events);
  stream.addEventListener("comment", (event) => {
    const comment = JSON.parse(event.data);
    const target = document.getElementById(comment.target);
    if (!target || document.getElementById(`c${comment.number}`)) return;
    target.insertAdjacentHTML("beforeend", comment.html);
    document.getElementById(`c${comment.number}`)?.classList.add("arrived");
    document.getElementById("no-comments")?.remove();
    document.getElementById("comment-count").textContent = comment.heading;
  });
})();
