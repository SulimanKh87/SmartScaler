const $ = (s) => document.querySelector(s);
let pollTimer = null, lastLogSize = 0;

function setProgress(pct){ $("#bar").style.width = `${Math.max(0, Math.min(100, pct))}%`; }

async function uploadFiles(){
  const files = $("#fileInput").files;
  if(!files || !files.length){ $("#uploadStatus").textContent = "No files selected."; return; }
  $("#uploadBtn").disabled = true; $("#uploadStatus").textContent = "Uploading...";
  const fd = new FormData(); for(const f of files) fd.append("files", f);
  try{
    const r = await fetch("/api/upload", { method:"POST", body:fd });
    const j = await r.json().catch(()=>({}));
    $("#uploadStatus").textContent = j.message || "Uploaded.";
  }catch(e){ $("#uploadStatus").textContent = "Upload error."; }
  $("#uploadBtn").disabled = false;
}

async function runResizer(){
  $("#resizerBtn").disabled = true; $("#resizerStatus").textContent = "Running resizer...";
  try{
    const r = await fetch("/api/run-resizer",{method:"POST"});
    const j = await r.json().catch(()=>({}));
    $("#resizerStatus").textContent = j.message || "Resizer complete.";
  }catch(e){ $("#resizerStatus").textContent = "Resizer error."; }
  $("#resizerBtn").disabled = false;
}

function startPolling(){
  if(pollTimer) clearInterval(pollTimer);
  pollTimer = null; lastLogSize = 0; $("#logBox").textContent = ""; setProgress(0);
  pollTimer = setInterval(async()=>{
    try{
      const s = await fetch("/api/status").then(r=>r.json());
      $("#runnerStatus").textContent = s.message || "";
      if(s.total && s.done >= 0) setProgress(Math.floor(100*s.done/s.total));
      const logs = await fetch(`/api/logs?offset=${lastLogSize}`).then(r=>r.json());
      if(logs && typeof logs.chunk === "string"){
        $("#logBox").textContent += logs.chunk; lastLogSize += logs.chunk.length; $("#logBox").scrollTop = $("#logBox").scrollHeight;
      }
      if(s.state === "idle"){ clearInterval(pollTimer); pollTimer = null; await refreshResults(); $("#runBtn").disabled = false; }
    }catch(e){}
  }, 700);
}

async function runUpscale(){
  $("#runBtn").disabled = true; $("#runnerStatus").textContent = "Starting..."; startPolling();
  const mode = document.querySelector('input[name="mode"]:checked').value;
  const r = await fetch("/api/run-upscale",{method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({mode})});
  const j = await r.json().catch(()=>({}));
  if(j.error){ $("#runnerStatus").innerHTML = `<span class="error">${j.error}</span>`; if(pollTimer){ clearInterval(pollTimer); pollTimer=null; } $("#runBtn").disabled=false; }
  else { $("#runnerStatus").textContent = "Running..."; }
}

async function refreshResults(){
  const r = await fetch("/api/list-results").then(r=>r.json());
  const out = await fetch("/api/out-location").then(r=>r.json()).catch(()=>({}));
  if(out && out.url) $("#openOut").innerHTML = `<a class="link" href="${out.url}" target="_blank">Open output folder</a>`;
  const box = $("#results"); box.innerHTML = "";
  if(r.csv){ const a = document.createElement("a"); a.href=r.csv; a.textContent="Download CSV"; a.className="link"; a.style.display="inline-block"; a.style.marginBottom="8px"; box.appendChild(a); }
  if(r.images && r.images.length){
    const grid = document.createElement("div");
    for(const img of r.images){
      const card = document.createElement("div"); card.className="thumb";
      const tag = document.createElement("img"); tag.src = img.url;
      const cap = document.createElement("span"); cap.textContent = img.name;
      card.appendChild(tag); card.appendChild(cap); grid.appendChild(card);
    }
    box.appendChild(grid);
  }
}

document.addEventListener("DOMContentLoaded",()=>{
  $("#uploadBtn").addEventListener("click", uploadFiles);
  $("#resizerBtn").addEventListener("click", runResizer);
  $("#runBtn").addEventListener("click", runUpscale);
  refreshResults();
});
