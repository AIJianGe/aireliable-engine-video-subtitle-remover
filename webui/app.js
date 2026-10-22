/* ============================================================
   VSR 视频字幕/水印移除 WebUI 前端逻辑
   1. 状态轮询（/status + /capabilities）
   2. 视频上传（POST /upload）
   3. 算法/OCR/选区参数
   4. SSE 推理（POST /infer）进度渲染 + 结果预览下载
   ============================================================ */

"use strict";

const $ = (id) => document.getElementById(id);

// ---------- 状态 ----------
let videoPath = null;      // 引擎侧视频绝对路径
let processing = false;    // 处理中标志
let statePollTimer = null;

// ---------- 原视频预览 / 结果队列状态 ----------
let sourceVideoUrl = null; // 原视频 /output URL（上传成功即可播放）
let results = [];          // 历史结果 [{id, video_url, variant, inpaint_mode, ...}]
let activeResultId = null; // 主视图当前展示的结果 id
let resultSeq = 0;         // 结果自增序号

// ---------- 首帧选区状态 ----------
let frameImage = null;     // 首帧 Image 对象
let frameNo = 0;           // 当前显示帧号
let imgRect = null;        // 图片在 canvas 内的显示区 {x, y, w, h}
let regions = [];          // 归一化选区 [{ymin, ymax, xmin, xmax}, ...]
let drag = null;           // 正在拖拽 {x1, y1, x2, y2}（canvas 逻辑坐标）
const FRAME_CANVAS_W = 860;
const FRAME_CANVAS_H = 484;

// ---------- 工具 ----------
function setBadge(state, detail) {
  const badge = $("state-badge");
  const clsMap = { ready: "ready", inferring: "inferring", loading: "loading", error: "error" };
  badge.className = "badge" + (clsMap[state] ? " " + clsMap[state] : "");
  badge.textContent = { ready: "就绪", loading: "加载中", inferring: "处理中", error: "异常", idle: "空闲", "": "未连接" }[state] || state || "未连接";
  $("state-detail").textContent = detail || "—";
}

function showError(msg) {
  $("error-text").textContent = msg;
  $("error-box").style.display = "block";
}

function hideError() {
  $("error-box").style.display = "none";
}

function setProgress(progress, text) {
  const wrap = $("progress-wrap");
  if (progress === null) {
    wrap.style.display = "none";
    return;
  }
  wrap.style.display = "flex";
  $("progress-fill").style.width = Math.round(progress * 100) + "%";
  $("progress-text").textContent = text || Math.round(progress * 100) + "%";
}

// ---------- 状态刷新 ----------
async function refreshStatus() {
  try {
    const res = await fetch("/status");
    if (!res.ok) { setBadge("", "引擎未响应"); return; }
    const st = await res.json();
    const extra = st.extra || {};
    window.__loadedVariant = st.loaded_variant || null;
    setBadge(st.state, `${st.loaded_variant || "未加载"} · ${extra.device_name || extra.device || ""}`);
  } catch (e) {
    setBadge("", "连接失败");
  }
}

// ---------- 算法对比说明（与 manifest variants 语义一致） ----------
const VARIANT_INFO = {
  "VSR/STTN-AUTO": {
    work: "时空 Transformer 视频修复，按你框选的区域直接生成遮罩逐帧擦除",
    ocr: "不需要",
    area: "必须框选区域",
    vram: "4GB",
    scene: "字幕/水印位置固定。质量高、时序连贯，默认推荐",
  },
  "VSR/STTN-DET": {
    work: "先 OCR 自动检测字幕位置，再用 STTN 视频修复擦除",
    ocr: "需要",
    area: "可不框选（OCR 自动定位）",
    vram: "4GB",
    scene: "字幕位置不固定、不想手动框选时用；检测不到的角落水印可再框选补充",
  },
  "VSR/LAMA": {
    work: "图像级 inpainting，逐帧独立修复（不利用前后帧时序）",
    ocr: "需要",
    area: "可不框选（OCR 自动定位）",
    vram: "2GB",
    scene: "显存最低、处理快；运动画面/长视频可能有轻微闪烁，适合静态字幕",
  },
  "VSR/PROPAINTER": {
    work: "视频级 inpainting，ProPainter + 光流传播，全片时序连贯修复",
    ocr: "需要",
    area: "可不框选（OCR 自动定位）",
    vram: "25GB",
    scene: "最高质量、最连贯；720p 需约 25G 显存，显存充足且追求画质时用",
  },
};

function fillVariantCompare() {
  const tbody = $("variant-compare-body");
  if (!tbody) return;
  tbody.innerHTML = "";
  const variants = window.__variants || [];
  const names = variants.length
    ? variants.map(v => v.name)
    : Object.keys(VARIANT_INFO);
  for (const name of names) {
    const info = VARIANT_INFO[name];
    if (!info) continue;
    const tr = document.createElement("tr");
    tr.className = "variant-row";
    tr.dataset.variant = name;
    const tag = name.split("/")[1] || name;
    tr.innerHTML =
      `<td class="v-name">${tag}</td>` +
      `<td>${info.work}</td>` +
      `<td class="${info.ocr === "需要" ? "v-need" : "v-none"}">${info.ocr}</td>` +
      `<td>${info.area}</td>` +
      `<td class="v-vram">${info.vram}</td>` +
      `<td>${info.scene}</td>`;
    tbody.appendChild(tr);
  }
}

// ---------- 能力加载（variant 列表） ----------
async function loadCapabilities() {
  try {
    const res = await fetch("/capabilities");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const cap = await res.json();
    const sel = $("variant-select");
    sel.innerHTML = "";
    const variants = cap.variants || [];
    window.__variants = variants;
    if (!variants.length) {
      sel.innerHTML = '<option value="">（无可用模型，请先到模型管理下载）</option>';
      return;
    }
    const defaultVariant = cap.default_variant || variants[0].name;
    for (const v of variants) {
      const opt = document.createElement("option");
      opt.value = v.name;
      const tag = v.requires_ocr ? " · 需OCR" : "";
      opt.textContent = `${v.name}（${v.vram_required || "?"}${tag}）`;
      opt.title = (VARIANT_INFO[v.name] || {}).scene || v.name;
      if (v.name === defaultVariant) opt.selected = true;
      sel.appendChild(opt);
    }
    fillVariantCompare();
    onVariantChange();
  } catch (e) {
    $("variant-select").innerHTML = '<option value="">（获取失败）</option>';
    showError("获取能力列表失败: " + e.message);
  }
}

function onVariantChange() {
  const vname = $("variant-select").value;
  // 从能力列表反查 requires_ocr
  const variant = (window.__variants || []).find(v => v.name === vname);
  const requiresOcr = variant ? variant.requires_ocr : vname !== "VSR/STTN-AUTO";
  $("detect-mode-row").style.display = requiresOcr ? "flex" : "none";
  // 高亮对比表中当前选中的算法行
  document.querySelectorAll("#variant-compare-body .variant-row").forEach(tr => {
    tr.classList.toggle("active", tr.dataset.variant === vname);
  });
  const info = VARIANT_INFO[vname];
  $("variant-desc").textContent = info
    ? `【推荐】${info.scene}${info.ocr === "需要" ? "；OCR 检测不到的字幕可在下方手动框选补充" : ""}`
    : (requiresOcr
        ? "该算法通过 OCR 自动检测字幕位置（STTN-AUTO 按选区直接擦除，无需 OCR）"
        : "STTN-AUTO 智能擦除：直接按下方选区生成遮罩，无需 OCR 检测");
  window.__selectedVariant = vname;
}

// ---------- 视频上传 ----------
function initUpload() {
  const dz = $("dropzone");
  const input = $("video-input");

  dz.addEventListener("click", () => input.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    if (e.dataTransfer.files.length) uploadVideo(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => {
    if (input.files.length) uploadVideo(input.files[0]);
  });
  $("btn-remove-video").addEventListener("click", () => {
    videoPath = null;
    sourceVideoUrl = null;
    input.value = "";
    $("source-video").removeAttribute("src");
    $("source-preview").style.display = "none";
    $("video-chip").style.display = "none";
    dz.style.display = "block";
    clearFrame();
  });
}

async function uploadVideo(file) {
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  $("video-name").textContent = file.name;
  $("video-size").textContent = file.size > 1048576 ? (file.size / 1048576).toFixed(1) + "MB" : Math.round(file.size / 1024) + "KB";
  $("video-chip").style.display = "inline-flex";
  $("dropzone").style.display = "none";
  showError(""); hideError();

  try {
    const res = await fetch("/upload", { method: "POST", body: fd });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    videoPath = data.path;
    // 原视频预览：优先用后端返回的 url（/output/uploads/{name}）；
    // 旧版本后端未返回 url 时，从绝对路径末段构造同样的 /output URL 兜底。
    sourceVideoUrl = data.url ||
      (data.path ? "/output/uploads/" + encodeURIComponent(String(data.path).split(/[\\/]/).pop()) : null);
    if (sourceVideoUrl) {
      $("source-video").src = sourceVideoUrl;
      $("source-preview").style.display = "block";
    } else {
      $("source-preview").style.display = "none";
    }
    // 上传成功后自动提取首帧供框选（失败不阻塞，可继续用预设按钮）
    clearFrame();
    await loadFirstFrame(videoPath, 0);
  } catch (e) {
    showError("视频上传失败: " + e.message);
    videoPath = null;
    sourceVideoUrl = null;
    $("source-video").removeAttribute("src");
    $("source-preview").style.display = "none";
    $("video-chip").style.display = "none";
    $("dropzone").style.display = "block";
  }
}

// ---------- 首帧选区（canvas 框选，支持多区域） ----------
function getCanvas() {
  return $("frame-canvas");
}

function drawFrame() {
  const canvas = getCanvas();
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!frameImage) return;

  // contain 布局：图片等比缩放居中
  const scale = Math.min(canvas.width / frameImage.width, canvas.height / frameImage.height);
  const w = frameImage.width * scale;
  const h = frameImage.height * scale;
  imgRect = { x: (canvas.width - w) / 2, y: (canvas.height - h) / 2, w, h };
  ctx.drawImage(frameImage, imgRect.x, imgRect.y, imgRect.w, imgRect.h);

  // 已选区域高亮（半透明填充 + 边框 + 序号）
  const colors = [
    "rgba(26,86,219,0.28)", "rgba(130,80,223,0.28)", "rgba(26,127,55,0.28)",
    "rgba(207,34,46,0.28)", "rgba(154,103,0,0.28)",
  ];
  regions.forEach((r, i) => {
    const x = imgRect.x + r.xmin * imgRect.w;
    const y = imgRect.y + r.ymin * imgRect.h;
    const rw = (r.xmax - r.xmin) * imgRect.w;
    const rh = (r.ymax - r.ymin) * imgRect.h;
    ctx.fillStyle = colors[i % colors.length];
    ctx.fillRect(x, y, rw, rh);
    ctx.strokeStyle = "#1a56db";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, rw, rh);
    ctx.fillStyle = "#1a56db";
    ctx.font = "bold 13px sans-serif";
    ctx.fillText(String(i + 1), x + 4, y + 16);
  });

  // 拖拽预览虚线框
  if (drag) {
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "rgba(26,86,219,0.9)";
    ctx.lineWidth = 2;
    ctx.strokeRect(
      Math.min(drag.x1, drag.x2),
      Math.min(drag.y1, drag.y2),
      Math.abs(drag.x2 - drag.x1),
      Math.abs(drag.y2 - drag.y1)
    );
    ctx.setLineDash([]);
  }
}

function mouseToCanvas(e) {
  const canvas = getCanvas();
  const rect = canvas.getBoundingClientRect();
  // 用逻辑尺寸比例换算，避免高分屏 / CSS 缩放误差
  return {
    x: (e.clientX - rect.left) * (canvas.width / rect.width),
    y: (e.clientY - rect.top) * (canvas.height / rect.height),
  };
}

function normalizeRegion(p1, p2) {
  if (!imgRect) return null;
  const x1 = Math.max(Math.min(p1.x, p2.x), imgRect.x);
  const x2 = Math.min(Math.max(p1.x, p2.x), imgRect.x + imgRect.w);
  const y1 = Math.max(Math.min(p1.y, p2.y), imgRect.y);
  const y2 = Math.min(Math.max(p1.y, p2.y), imgRect.y + imgRect.h);
  if (x2 - x1 < 3 || y2 - y1 < 3) return null;   // 过滤误点
  return {
    xmin: (x1 - imgRect.x) / imgRect.w,
    ymin: (y1 - imgRect.y) / imgRect.h,
    xmax: (x2 - imgRect.x) / imgRect.w,
    ymax: (y2 - imgRect.y) / imgRect.h,
  };
}

function addRegion(region) {
  if (!region) return;
  regions.push(region);
  renderRegionList();
  drawFrame();
}

function removeRegion(idx) {
  if (idx < 0 || idx >= regions.length) return;
  regions.splice(idx, 1);
  renderRegionList();
  drawFrame();
}

function undoRegion() {
  if (!regions.length) return;
  regions.pop();
  renderRegionList();
  drawFrame();
}

function clearRegions() {
  regions = [];
  renderRegionList();
  drawFrame();
}

function renderRegionList() {
  const list = $("region-list");
  list.innerHTML = "";
  $("region-count").textContent = `已选 ${regions.length} 个区域`;
  regions.forEach((r, i) => {
    const item = document.createElement("div");
    item.className = "region-item";
    const label = document.createElement("span");
    label.textContent =
      `区域${i + 1}: ymin=${r.ymin.toFixed(3)} ymax=${r.ymax.toFixed(3)} ` +
      `xmin=${r.xmin.toFixed(3)} xmax=${r.xmax.toFixed(3)}`;
    const rm = document.createElement("button");
    rm.className = "region-rm";
    rm.textContent = "×";
    rm.title = "删除该区域";
    rm.addEventListener("click", () => removeRegion(i));
    item.appendChild(label);
    item.appendChild(rm);
    list.appendChild(item);
  });
}

async function loadFirstFrame(path, frameIdx) {
  const canvas = getCanvas();
  const ph = $("frame-placeholder");
  try {
    ph.style.display = "flex";
    ph.innerHTML = "<p>正在提取首帧…</p>";
    const res = await fetch("/first_frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_path: path, frame_no: frameIdx }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error((data.detail && data.detail.message) || (data.detail) || "HTTP " + res.status);
    }
    const data = await res.json();
    const img = new Image();
    img.onload = () => {
      frameImage = img;
      frameNo = data.frame_no != null ? data.frame_no : frameIdx;
      $("frame-no-input").value = frameNo;
      ph.style.display = "none";
      canvas.style.display = "block";
      $("region-toolbar").style.display = "flex";
      drawFrame();
    };
    img.onerror = () => {
      ph.innerHTML = "<p>首帧图片解码失败</p><p class='dz-sub'>可继续使用预设按钮选择区域</p>";
      showError("首帧图片解码失败");
    };
    img.src = data.image;
  } catch (e) {
    ph.innerHTML = "<p>首帧提取失败</p><p class='dz-sub'>可继续使用预设按钮选择区域</p>";
    showError("首帧提取失败: " + e.message);
  }
}

function clearFrame() {
  frameImage = null;
  frameNo = 0;
  imgRect = null;
  regions = [];
  drag = null;
  const canvas = getCanvas();
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  canvas.style.display = "none";
  $("frame-placeholder").style.display = "flex";
  $("frame-placeholder").innerHTML =
    "<p>上传视频后自动显示首帧</p>" +
    "<p class='dz-sub'>按住鼠标左键在画面上拖拽，框选要移除的字幕/水印区域（可多个）</p>";
  $("region-toolbar").style.display = "none";
  renderRegionList();
}

function initRegionEditor() {
  // 预设按钮 → 追加一个区域（不替换已有区域）
  document.querySelectorAll("#area-presets .preset").forEach(p => {
    p.addEventListener("click", () => {
      const parts = p.dataset.area.split(",").map(Number);
      addRegion({ ymin: parts[0], ymax: parts[1], xmin: parts[2], xmax: parts[3] });
    });
  });

  // canvas 鼠标框选
  const canvas = getCanvas();
  canvas.addEventListener("mousedown", (e) => {
    if (!frameImage) return;
    const p = mouseToCanvas(e);
    drag = { x1: p.x, y1: p.y, x2: p.x, y2: p.y };
    drawFrame();
  });
  canvas.addEventListener("mousemove", (e) => {
    if (!drag) return;
    const p = mouseToCanvas(e);
    drag.x2 = p.x;
    drag.y2 = p.y;
    drawFrame();
  });
  canvas.addEventListener("mouseup", (e) => {
    if (!drag) return;
    const p = mouseToCanvas(e);
    drag.x2 = p.x;
    drag.y2 = p.y;
    const region = normalizeRegion({ x: drag.x1, y: drag.y1 }, { x: drag.x2, y: drag.y2 });
    drag = null;
    drawFrame();
    if (region) addRegion(region);
  });
  canvas.addEventListener("mouseleave", () => {
    if (drag) { drag = null; drawFrame(); }
  });

  // 工具栏
  $("btn-undo-region").addEventListener("click", undoRegion);
  $("btn-clear-regions").addEventListener("click", clearRegions);
  $("btn-refresh-frame").addEventListener("click", () => {
    if (!videoPath) return;
    const idx = Math.max(0, parseInt($("frame-no-input").value, 10) || 0);
    loadFirstFrame(videoPath, idx);
  });
}

function parseAbSections() {
  const raw = $("ab-input").value.trim();
  if (!raw) return null;
  const sections = [];
  for (const seg of raw.split(/[,，;；]/)) {
    const m = seg.trim().match(/^(\d+)\s*[-~—]\s*(\d+)$/);
    if (!m) continue;
    const a = parseInt(m[1]), b = parseInt(m[2]);
    if (b > a) sections.push([a, b]);
  }
  return sections.length ? sections : null;
}

// ---------- 加载/卸载 ----------
async function doLoad() {
  const variant = $("variant-select").value;
  if (!variant) { showError("请先选择算法（variant）"); return; }
  setBadge("loading", "加载中…");
  try {
    const res = await fetch("/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ variant }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error((data.error && (data.error.message || data.error.detail)) || "HTTP " + res.status);
    // 轮询 /status 直到 ready / error
    await pollUntilReady();
  } catch (e) {
    setBadge("error", "加载失败");
    showError("加载失败: " + e.message);
  }
}

async function pollUntilReady(timeoutMs = 60000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    await new Promise(r => setTimeout(r, 1200));
    try {
      const res = await fetch("/status");
      const st = await res.json();
      if (st.state === "ready") { setBadge("ready", "加载完成"); return; }
      if (st.state === "error") { setBadge("error", "加载失败"); return; }
      setBadge("loading", "加载中…");
    } catch (e) { /* 忽略瞬时错误 */ }
  }
}

async function doUnload() {
  try {
    const res = await fetch("/unload", { method: "POST" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    setBadge("idle", "已卸载");
  } catch (e) {
    showError("卸载失败: " + e.message);
  }
}

// ---------- 推理（SSE） ----------
async function doProcess() {
  if (processing) return;
  const variant = $("variant-select").value;
  if (!variant) { showError("请先选择算法（variant）"); return; }
  if (!videoPath) { showError("请先上传目标视频"); return; }

  hideError();
  processing = true;
  $("btn-process").disabled = true;
  $("btn-load").disabled = true;
  $("btn-unload").disabled = true;
  setBadge("inferring", "处理中…");
  setProgress(0, "提交任务…");

  // 所选算法 ≠ 已加载模型时：先提示切换（引擎会处理前自动加载）
  if (variant && window.__loadedVariant && variant !== window.__loadedVariant) {
    setBadge("loading", "切换模型中…");
    setProgress(0, `正在切换模型（${window.__loadedVariant} → ${variant}）…`);
  }

  const body = {
    variant: variant,
    inputs: {
      video_path: videoPath,
      // 多区域：每个框选区域转 [ymin, ymax, xmin, xmax]；空数组 = 后端全屏兜底
      sub_areas: regions.map(r => [r.ymin, r.ymax, r.xmin, r.xmax]),
      ab_sections: parseAbSections(),
      detect_mode: $("detect-mode-select").value,
      params: {},
    },
  };

  try {
    const res = await fetch("/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) {
      const errText = await res.text();
      throw new Error("HTTP " + res.status + ": " + errText.slice(0, 300));
    }

    await consumeSSE(res.body, (frame) => {
      if (frame.stage === "error") {
        const err = frame.error || {};
        throw new Error(err.message || "处理失败");
      }
      if (frame.stage === "completed") {
        setProgress(1, "完成");
        renderResult(frame.result || {});
      } else if (typeof frame.progress === "number") {
        // 切换模型完成的后续帧（start/preparing/processing）到达 → 徽标恢复"处理中"
        if (frame.stage !== "model_loading" && frame.stage !== "video_subtitle.heartbeat") {
          setBadge("inferring", "处理中…");
        }
        setProgress(frame.progress, frame.message || "");
      }
    });

    refreshStatus();
  } catch (e) {
    setBadge("error", "处理失败");
    showError(e.message || "处理失败");
  } finally {
    processing = false;
    $("btn-process").disabled = false;
    $("btn-load").disabled = false;
    $("btn-unload").disabled = false;
    setProgress(null);
    refreshStatus();
  }
}

async function consumeSSE(body, onFrame) {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      for (const line of chunk.split("\n")) {
        if (line.startsWith("data:")) {
          const data = line.slice(5).trim();
          if (!data) continue;
          try {
            onFrame(JSON.parse(data));
          } catch (e) { /* 忽略无法解析的帧 */ }
        }
      }
    }
  }
}

// ---------- 结果渲染（主视图 + 历史结果排队列表） ----------
function formatClock(d) {
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

// SSE completed 帧回调：把结果追加进历史队列，并默认展示最新一条
function renderResult(result) {
  resultSeq += 1;
  const entry = {
    id: resultSeq,
    video_url: result.video_url || "",
    inpaint_mode: result.inpaint_mode || "",
    variant: result.variant || "",
    detect_mode: result.detect_mode || "",
    elapsed: result.elapsed != null ? result.elapsed + "s" : "",
    input_video: result.input_video || "",
    time: new Date(),
  };
  results.push(entry);
  activeResultId = entry.id;
  showResultInMain(entry);
  renderResultList();
  $("result-list-head").style.display = "flex";
}

// 把指定结果渲染到主视图（大播放器 + meta）
function showResultInMain(entry) {
  const view = $("result-view");
  view.innerHTML = "";
  if (!entry.video_url) {
    view.innerHTML = '<p class="placeholder">未返回视频结果</p>';
  } else {
    const label = document.createElement("div");
    label.className = "media-label";
    label.textContent = "处理结果（已保留原音频） · " + formatClock(entry.time);
    const video = document.createElement("video");
    video.src = entry.video_url;
    video.controls = true;
    video.autoplay = false;
    video.preload = "metadata";
    view.appendChild(label);
    view.appendChild(video);
  }

  const meta = $("result-meta");
  meta.innerHTML = "";
  const chips = [];
  if (entry.inpaint_mode) chips.push(["算法", entry.inpaint_mode]);
  if (entry.variant) chips.push(["variant", entry.variant]);
  if (entry.detect_mode) chips.push(["OCR", entry.detect_mode]);
  if (entry.elapsed) chips.push(["耗时", entry.elapsed]);
  if (entry.input_video) chips.push(["输入", entry.input_video]);
  if (entry.video_url) {
    chips.push(["下载", `<a href="${entry.video_url}" download>点此下载</a>`]);
  }
  for (const [k, v] of chips) {
    const c = document.createElement("span");
    c.className = "meta-chip";
    c.innerHTML = `${k}: ${v}`;
    meta.appendChild(c);
  }
}

// 渲染历史结果列表（每次处理依次追加，最新一条默认高亮）
function renderResultList() {
  const list = $("result-list");
  const head = $("result-list-head");
  list.innerHTML = "";

  if (!results.length) {
    head.style.display = "none";
    return;
  }
  head.style.display = "flex";
  $("result-list-count").textContent = results.length;

  results.forEach((r, i) => {
    const card = document.createElement("div");
    card.className = "result-card" + (r.id === activeResultId ? " active" : "");

    // 缩略播放器（preload=metadata 轻量加载首帧）
    const video = document.createElement("video");
    video.src = r.video_url;
    video.controls = true;
    video.preload = "metadata";
    video.playsInline = true;

    // 信息区
    const info = document.createElement("div");
    info.className = "result-card-info";
    const title = document.createElement("div");
    title.className = "result-card-title";
    title.textContent = `结果 ${i + 1} · ${r.input_video || "视频"} · ${formatClock(r.time)}`;
    const meta = document.createElement("div");
    meta.className = "result-card-meta";
    for (const t of [r.inpaint_mode, r.variant, r.detect_mode && ("OCR:" + r.detect_mode), r.elapsed]) {
      if (!t) continue;
      const s = document.createElement("span");
      s.className = "meta-chip";
      s.textContent = t;
      meta.appendChild(s);
    }
    info.appendChild(title);
    info.appendChild(meta);

    // 操作区
    const actions = document.createElement("div");
    actions.className = "result-card-actions";
    const btnView = document.createElement("button");
    btnView.className = "btn btn-secondary btn-sm";
    btnView.textContent = "查看";
    btnView.addEventListener("click", () => {
      activeResultId = r.id;
      showResultInMain(r);
      renderResultList();
    });
    const btnDl = document.createElement("a");
    btnDl.className = "btn btn-ghost btn-sm";
    btnDl.href = r.video_url;
    btnDl.download = "";
    btnDl.textContent = "下载";
    const btnRm = document.createElement("button");
    btnRm.className = "btn btn-danger btn-sm";
    btnRm.textContent = "删除";
    btnRm.addEventListener("click", () => {
      results.splice(results.findIndex(x => x.id === r.id), 1);
      if (activeResultId === r.id) {
        if (results.length) {
          const last = results[results.length - 1];
          activeResultId = last.id;
          showResultInMain(last);
        } else {
          activeResultId = null;
          resetResultMain();
        }
      }
      renderResultList();
    });
    actions.appendChild(btnView);
    actions.appendChild(btnDl);
    actions.appendChild(btnRm);

    card.appendChild(video);
    card.appendChild(info);
    card.appendChild(actions);
    list.appendChild(card);
  });
}

function resetResultMain() {
  $("result-view").innerHTML = '<p class="placeholder">处理后的视频将显示在这里</p>';
  $("result-meta").innerHTML = "";
}

function clearResults() {
  results = [];
  activeResultId = null;
  resetResultMain();
  renderResultList();
}

// ---------- 算法对比悬浮窗口（CSS :hover 桥接 + JS mouseenter 双保险） ----------
// 主机制（CSS）：问号图标 + 弹窗共用 .variant-help 容器，弹窗是容器子元素，
// 鼠标从图标滑入弹窗全程不离开容器 → .variant-help:hover 恒真 → 弹窗不提前关闭。
// 双保险（JS）：在容器上挂 mouseenter/mouseleave 显式维护 .open 类，
// 与 CSS :hover 驱动同一显示规则。弹窗是容器 DOM 子元素，移入弹窗不会触发
// mouseleave，所以桥接语义与 CSS 完全一致；同时不依赖 :hover 匹配的浏览器差异。
// 触屏/无 hover 环境：matchMedia(hover:hover) 为 false → 改用点击切换。
function initVariantHelp() {
  const help = $("variant-help");
  const icon = $("variant-help-icon");
  if (!help || !icon) return;

  const canHover = !window.matchMedia || window.matchMedia("(hover: hover)").matches;

  function setOpen(open) {
    help.classList.toggle("open", open);
    icon.setAttribute("aria-expanded", String(open));
  }

  if (canHover) {
    // 悬停环境：进入容器（图标或弹窗）→ 开；离开容器 → 关。
    // 鼠标从图标滑入弹窗全程都在容器内，不会触发 mouseleave。
    help.addEventListener("mouseenter", () => setOpen(true));
    help.addEventListener("mouseleave", () => {
      setOpen(false);
      // 点击图标会给它焦点，若不主动失焦，键盘焦点会持续保留在图标上。
      // CSS 显示规则已不含 :focus-within，这里主动 blur 防止焦点残留引发状态混乱。
      if (document.activeElement === icon) icon.blur();
    });
    // 键盘/点击焦点：图标获得焦点 → 开（替代 :focus-within 的键盘支持）；
    // 失焦 → 关。点击图标时 focus 先触发，弹窗打开；随后点击不干扰开关。
    icon.addEventListener("focus", () => setOpen(true));
    icon.addEventListener("blur", () => setOpen(false));
    // 悬停已接管开关，点击图标不干扰（避免 hover 打开后点击又关闭的困惑）
    icon.addEventListener("click", (e) => e.stopPropagation());
    // 键盘 Enter/Space 明确打开（关闭交给 blur / mouseleave）
    icon.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        setOpen(true);
      }
    });
  } else {
    // 触屏/无 hover 环境：点击图标切换弹窗，点击页面其它处关闭。
    icon.addEventListener("click", (e) => {
      e.stopPropagation();
      setOpen(!help.classList.contains("open"));
    });
    document.addEventListener("click", (e) => {
      if (help.classList.contains("open") && !help.contains(e.target)) {
        setOpen(false);
      }
    });
    // 键盘 Enter/Space 等价于点击（切换开关）
    icon.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        icon.click();
      }
    });
  }
}

// ---------- 初始化 ----------
function init() {
  initUpload();
  initRegionEditor();
  initVariantHelp();

  $("variant-select").addEventListener("change", onVariantChange);
  $("btn-load").addEventListener("click", doLoad);
  $("btn-unload").addEventListener("click", doUnload);
  $("btn-process").addEventListener("click", doProcess);
  $("btn-refresh-status").addEventListener("click", refreshStatus);
  $("btn-clear-results").addEventListener("click", clearResults);

  // 先刷新状态，再加载能力
  refreshStatus();
  loadCapabilities().then(() => refreshStatus());
  // 定期刷新状态（非处理中）
  setInterval(() => { if (!processing) refreshStatus(); }, 15000);
}

document.addEventListener("DOMContentLoaded", init);
