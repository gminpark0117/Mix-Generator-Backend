from __future__ import annotations

from fastapi.responses import HTMLResponse

from atomix.main import app


MOCK_FRONTEND_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Atomix Mock Frontend</title>
  <style>
    :root {
      --bg: #f8fafc;
      --panel: #ffffff;
      --line: #cbd5e1;
      --text: #0f172a;
      --subtle: #475569;
      --ok: #065f46;
      --error: #991b1b;
      --accent: #0f766e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #f8fafc 0%, #ecfeff 100%);
    }
    main {
      max-width: 980px;
      margin: 0 auto;
      padding: 24px 16px 40px;
    }
    h1 { margin: 0 0 8px; font-size: 1.8rem; }
    p { margin: 0 0 16px; color: var(--subtle); }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 14px;
      margin-bottom: 14px;
    }
    .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 10px; }
    label { font-size: 0.92rem; display: flex; flex-direction: column; gap: 4px; }
    input[type="text"], input[type="number"] {
      min-width: 220px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      font: inherit;
    }
    input[type="file"] { font: inherit; }
    button {
      border: 0;
      border-radius: 8px;
      padding: 10px 12px;
      background: var(--accent);
      color: #fff;
      font-weight: 600;
      cursor: pointer;
    }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 0.9rem;
    }
    th, td {
      border: 1px solid var(--line);
      padding: 6px 8px;
      text-align: left;
    }
    th { background: #f1f5f9; }
    .track-meta-row {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
      margin-bottom: 8px;
    }
    .status { margin-top: 8px; font-weight: 600; }
    .status.ok { color: var(--ok); }
    .status.error { color: var(--error); }
    pre {
      margin: 0;
      overflow: auto;
      max-height: 260px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #f8fafc;
      font-size: 0.85rem;
    }
    .audio-box { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    @media (max-width: 700px) {
      .track-meta-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <h1>Atomix Mock Frontend</h1>
    <p>Upload tracks to test <code>create_mix</code> and <code>add_tracks_to_mix</code> quickly.</p>

    <section>
      <div class="row">
        <label>API prefix
          <input id="apiPrefix" type="text" value="/v1" />
        </label>
      </div>
    </section>

    <section>
      <h2>Create Mix</h2>
      <div class="row">
        <input id="createFiles" type="file" accept="audio/*" multiple />
        <button id="createBtn">Create Mix</button>
      </div>
      <div id="createMeta"></div>
    </section>

    <section>
      <h2>Add Tracks To Mix</h2>
      <div class="row">
        <label>Mix ID
          <input id="mixId" type="text" placeholder="Paste mix_id or create mix first" />
        </label>
        <label>Client playhead ms
          <input id="playheadMs" type="number" min="0" value="0" />
        </label>
      </div>
      <div class="row">
        <input id="addFiles" type="file" accept="audio/*" multiple />
        <button id="addBtn">Add Tracks</button>
      </div>
      <div id="addMeta"></div>
    </section>

    <section>
      <h2>Latest Response</h2>
      <div id="status" class="status"></div>
      <pre id="jsonOut">{}</pre>
    </section>

    <section>
      <h2>Tracklist</h2>
      <div id="tracklistOut">No tracklist yet.</div>
    </section>

    <section>
      <h2>Rendered Audio</h2>
      <div class="audio-box">
        <audio id="audioPlayer" controls></audio>
        <a id="downloadLink" href="#" download hidden>Download audio</a>
      </div>
    </section>
  </main>

  <script>
    const createFiles = document.getElementById("createFiles");
    const addFiles = document.getElementById("addFiles");
    const createMeta = document.getElementById("createMeta");
    const addMeta = document.getElementById("addMeta");
    const createBtn = document.getElementById("createBtn");
    const addBtn = document.getElementById("addBtn");
    const statusEl = document.getElementById("status");
    const jsonOut = document.getElementById("jsonOut");
    const tracklistOut = document.getElementById("tracklistOut");
    const audioPlayer = document.getElementById("audioPlayer");
    const downloadLink = document.getElementById("downloadLink");
    const mixIdInput = document.getElementById("mixId");
    const playheadInput = document.getElementById("playheadMs");
    const apiPrefixInput = document.getElementById("apiPrefix");

    function stripExt(name) {
      const i = name.lastIndexOf(".");
      return i > 0 ? name.slice(0, i) : name;
    }

    function buildMetaInputs(files, container) {
      container.innerHTML = "";
      if (!files.length) {
        return;
      }
      files.forEach((file, idx) => {
        const row = document.createElement("div");
        row.className = "track-meta-row";

        const order = document.createElement("input");
        order.type = "text";
        order.value = "#" + (idx + 1);
        order.readOnly = true;

        const song = document.createElement("input");
        song.type = "text";
        song.placeholder = "song_name";
        song.value = stripExt(file.name);
        song.dataset.field = "song_name";

        const artist = document.createElement("input");
        artist.type = "text";
        artist.placeholder = "artist_name";
        artist.value = "Unknown Artist";
        artist.dataset.field = "artist_name";

        row.appendChild(order);
        row.appendChild(song);
        row.appendChild(artist);
        container.appendChild(row);
      });
    }

    function collectMetadata(container) {
      const rows = Array.from(container.querySelectorAll(".track-meta-row"));
      return rows.map((row) => {
        const inputs = row.querySelectorAll("input");
        return {
          song_name: inputs[1].value || "Unknown Song",
          artist_name: inputs[2].value || "Unknown Artist",
        };
      });
    }

    function setStatus(message, ok) {
      statusEl.textContent = message;
      statusEl.className = ok ? "status ok" : "status error";
    }

    function renderTracklist(tracklist) {
      if (!Array.isArray(tracklist) || tracklist.length === 0) {
        tracklistOut.textContent = "No tracklist in response.";
        return;
      }
      const headers = ["position", "song_name", "artist_name", "start_ms", "end_ms", "source_start_ms"];
      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const headRow = document.createElement("tr");
      headers.forEach((h) => {
        const th = document.createElement("th");
        th.textContent = h;
        headRow.appendChild(th);
      });
      thead.appendChild(headRow);
      table.appendChild(thead);

      const tbody = document.createElement("tbody");
      tracklist.forEach((row) => {
        const tr = document.createElement("tr");
        headers.forEach((h) => {
          const td = document.createElement("td");
          td.textContent = String(row[h] ?? "");
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      tracklistOut.innerHTML = "";
      tracklistOut.appendChild(table);
    }

    function renderAudio(payload) {
      const audioUrl = payload?.revision?.audio_url;
      if (!audioUrl) {
        audioPlayer.removeAttribute("src");
        downloadLink.hidden = true;
        return;
      }
      const resolved = new URL(audioUrl, window.location.origin).toString();
      audioPlayer.src = resolved;
      const revNo = payload?.revision?.revision_no ?? "x";
      downloadLink.href = resolved;
      downloadLink.download = "mix-" + (payload.mix_id || "unknown") + "-rev-" + revNo + ".wav";
      downloadLink.hidden = false;
    }

    async function submitMix(url, formData, loadingButton) {
      loadingButton.disabled = true;
      try {
        const res = await fetch(url, { method: "POST", body: formData });
        const body = await res.json().catch(() => ({}));
        jsonOut.textContent = JSON.stringify(body, null, 2);
        if (!res.ok) {
          const detail = body?.detail || "Request failed";
          setStatus(String(detail), false);
          renderTracklist([]);
          renderAudio(null);
          return null;
        }
        setStatus("Request succeeded.", true);
        renderTracklist(body.tracklist);
        renderAudio(body);
        return body;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setStatus(msg, false);
        return null;
      } finally {
        loadingButton.disabled = false;
      }
    }

    createFiles.addEventListener("change", () => buildMetaInputs(Array.from(createFiles.files || []), createMeta));
    addFiles.addEventListener("change", () => buildMetaInputs(Array.from(addFiles.files || []), addMeta));

    createBtn.addEventListener("click", async () => {
      const files = Array.from(createFiles.files || []);
      if (!files.length) {
        setStatus("Select at least one file for create_mix.", false);
        return;
      }
      const form = new FormData();
      files.forEach((f) => form.append("files", f));
      form.append("tracks_metadata", JSON.stringify(collectMetadata(createMeta)));
      const apiPrefix = (apiPrefixInput.value || "/v1").trim().replace(/\\/$/, "");
      const payload = await submitMix(apiPrefix + "/mixes", form, createBtn);
      if (payload?.mix_id) {
        mixIdInput.value = payload.mix_id;
      }
    });

    addBtn.addEventListener("click", async () => {
      const mixId = mixIdInput.value.trim();
      const files = Array.from(addFiles.files || []);
      if (!mixId) {
        setStatus("mix_id is required for add_tracks_to_mix.", false);
        return;
      }
      if (!files.length) {
        setStatus("Select at least one file for add_tracks_to_mix.", false);
        return;
      }
      const form = new FormData();
      form.append("client_playhead_ms", String(Number(playheadInput.value || 0)));
      files.forEach((f) => form.append("files", f));
      form.append("tracks_metadata", JSON.stringify(collectMetadata(addMeta)));
      const apiPrefix = (apiPrefixInput.value || "/v1").trim().replace(/\\/$/, "");
      await submitMix(apiPrefix + "/mixes/" + encodeURIComponent(mixId) + "/tracks:upload", form, addBtn);
    });
  </script>
</body>
</html>
"""


@app.get("/mock_frontend", response_class=HTMLResponse)
async def mock_frontend() -> HTMLResponse:
    return HTMLResponse(content=MOCK_FRONTEND_HTML)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mock_frontend:app", host="127.0.0.1", port=8000, reload=True)
