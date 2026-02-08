#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const http = require("http");
const vm = require("vm");
const url = require("url");

const puppeteer = require("puppeteer");
const stylus = require("stylus");
const doT = require("dot");

const DEFAULT_KLE_ROOT = path.join(__dirname, "vendor", "keyboard-layout-editor");
const DEFAULT_VIEWPORT = { width: 5000, height: 4000 };
const DEFAULT_QUALITY = 92;
const DEFAULT_NAV_TIMEOUT_MS = 0;
const DEFAULT_WAIT_UNTIL = "domcontentloaded";
const DEFAULT_FONT_WAIT_MS = 2000;
const DEFAULT_RENDER_MODE = "screenshot";
const FA_TEST_GLYPH = "\uf04b";

function parseArgs(argv) {
  const args = {
    inputs: [],
    output: null,
    outDir: null,
    kleRoot: DEFAULT_KLE_ROOT,
    quality: DEFAULT_QUALITY,
    scale: 1,
    viewport: { ...DEFAULT_VIEWPORT },
    waitUntil: DEFAULT_WAIT_UNTIL,
    navTimeoutMs: DEFAULT_NAV_TIMEOUT_MS,
    fontWaitMs: DEFAULT_FONT_WAIT_MS,
    renderMode: DEFAULT_RENDER_MODE,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith("-")) {
      args.inputs.push(arg);
      continue;
    }

    switch (arg) {
      case "-o":
      case "--output":
        args.output = argv[++i];
        break;
      case "--out-dir":
        args.outDir = argv[++i];
        break;
      case "--kle-root":
        args.kleRoot = argv[++i];
        break;
      case "--quality":
        args.quality = Number(argv[++i]);
        break;
      case "--scale":
        args.scale = Number(argv[++i]);
        break;
      case "--viewport": {
        const raw = argv[++i];
        const match = /^([0-9]+)x([0-9]+)$/i.exec(raw || "");
        if (!match) {
          throw new Error("--viewport must be in the form WIDTHxHEIGHT, e.g. 5000x4000");
        }
        args.viewport = { width: Number(match[1]), height: Number(match[2]) };
        break;
      }
      case "--wait":
        args.waitUntil = argv[++i];
        break;
      case "--timeout":
        args.navTimeoutMs = Number(argv[++i]);
        break;
      case "--font-wait":
        args.fontWaitMs = Number(argv[++i]);
        break;
      case "--render-mode":
        args.renderMode = argv[++i];
        break;
      case "-h":
      case "--help":
        printHelp();
        process.exit(0);
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (!args.inputs.length) {
    throw new Error("No input JSON files or directories provided.");
  }

  if (args.output && args.inputs.length > 1) {
    throw new Error("--output can only be used with a single input file.");
  }

  if (!Number.isFinite(args.quality) || args.quality < 1 || args.quality > 95) {
    throw new Error("--quality must be between 1 and 95.");
  }

  if (!Number.isFinite(args.scale) || args.scale <= 0) {
    throw new Error("--scale must be a positive number.");
  }

  const waitChoices = new Set(["domcontentloaded", "load", "networkidle0", "networkidle2"]);
  if (!waitChoices.has(args.waitUntil)) {
    throw new Error("--wait must be one of: domcontentloaded, load, networkidle0, networkidle2");
  }

  if (!Number.isFinite(args.navTimeoutMs) || args.navTimeoutMs < 0) {
    throw new Error("--timeout must be a non-negative number (0 disables timeouts).");
  }

  if (!Number.isFinite(args.fontWaitMs) || args.fontWaitMs < 0) {
    throw new Error("--font-wait must be a non-negative number (0 disables waiting).");
  }

  const renderModeChoices = new Set(["screenshot", "kle-jpg"]);
  if (!renderModeChoices.has(args.renderMode)) {
    throw new Error("--render-mode must be one of: screenshot, kle-jpg");
  }

  return args;
}

function printHelp() {
  console.log(`\
Usage:
  node kle_render_jpg.js <input.json> [-o output.jpg]
  node kle_render_jpg.js <dir> --out-dir out/

Options:
  --kle-root <path>   Path to a local clone of keyboard-layout-editor
  --quality <1-95>    JPEG quality (default: ${DEFAULT_QUALITY})
  --scale <number>    Device scale factor for screenshot (default: 1)
  --viewport WxH      Browser viewport size (default: ${DEFAULT_VIEWPORT.width}x${DEFAULT_VIEWPORT.height})
  --wait <mode>       Wait mode for page content (default: ${DEFAULT_WAIT_UNTIL})
  --timeout <ms>      Navigation timeout in ms (default: ${DEFAULT_NAV_TIMEOUT_MS}, 0 = no timeout)
  --font-wait <ms>    Max time to wait for fonts (default: ${DEFAULT_FONT_WAIT_MS})
  --render-mode <m>   screenshot (default) or kle-jpg (html2canvas-based)
`);
}

function discoverInputs(inputs) {
  const results = [];
  for (const input of inputs) {
    const stat = fs.statSync(input);
    if (stat.isDirectory()) {
      const files = fs
        .readdirSync(input)
        .filter((name) => name.toLowerCase().endsWith(".json"))
        .map((name) => path.join(input, name));
      results.push(...files);
    } else {
      results.push(input);
    }
  }
  return results;
}

function extractTemplate(html, id) {
  const re = new RegExp(`(<script[^>]*id=["']${id}["'][^>]*>)([\\s\\S]*?)(</script>)`, "i");
  const match = html.match(re);
  if (!match) {
    throw new Error(`Unable to find template '${id}' in kb.html`);
  }
  return match[2];
}

function compileStylusFile(filePath) {
  const src = fs.readFileSync(filePath, "utf8");
  return new Promise((resolve, reject) => {
    stylus(src)
      .set("filename", filePath)
      .set("compress", true)
      .render((err, css) => {
        if (err) reject(err);
        else resolve(css);
      });
  });
}

function findFontAwesomePaths(kleRoot) {
  const candidates = [
    path.join(__dirname, "node_modules", "font-awesome"),
    path.join(process.cwd(), "node_modules", "font-awesome"),
  ];

  for (const base of candidates) {
    const css = path.join(base, "css", "font-awesome.min.css");
    const fonts = path.join(base, "fonts");
    if (fs.existsSync(css) && fs.existsSync(fonts)) {
      return { css, fonts };
    }
  }

  const legacyCss = path.join(kleRoot, "css", "font-awesome.min.css");
  const legacyFonts = path.join(kleRoot, "fonts");
  if (fs.existsSync(legacyCss)) {
    return { css: legacyCss, fonts: legacyFonts };
  }

  return null;
}

function resolveFontAwesome(kleRoot) {
  const paths = findFontAwesomePaths(kleRoot);
  if (!paths) {
    return null;
  }
  return {
    cssHref: "/fa/css/font-awesome.min.css",
    mountDir: path.dirname(paths.css),
    fontsDir: paths.fonts,
  };
}

function resolveHtml2Canvas() {
  const candidates = [
    path.join(__dirname, "node_modules", "html2canvas", "dist", "html2canvas.min.js"),
    path.join(process.cwd(), "node_modules", "html2canvas", "dist", "html2canvas.min.js"),
  ];
  for (const scriptPath of candidates) {
    if (fs.existsSync(scriptPath)) {
      return {
        scriptHref: "/deps/html2canvas.min.js",
        mountDir: path.dirname(scriptPath),
      };
    }
  }
  return null;
}

function startStaticServer(root, mounts = []) {
  const normalizedMounts = mounts.map((mount) => {
    const prefix = mount.prefix.endsWith("/") ? mount.prefix : `${mount.prefix}/`;
    return { prefix, dir: mount.dir };
  });

  const server = http.createServer((req, res) => {
    const parsed = url.parse(req.url || "");
    const pathname = decodeURIComponent(parsed.pathname || "/");

    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "*");
    if (req.method === "OPTIONS") {
      res.statusCode = 204;
      res.end();
      return;
    }

    let filePath = null;
    let baseDir = null;

    for (const mount of normalizedMounts) {
      if (pathname === mount.prefix.slice(0, -1) || pathname.startsWith(mount.prefix)) {
        const relative = pathname.slice(mount.prefix.length).replace(/^([/\\])+/, "");
        filePath = path.join(mount.dir, relative);
        baseDir = mount.dir;
        break;
      }
    }

    if (!filePath) {
      const safePath = path.normalize(pathname).replace(/^([/\\])+/, "");
      filePath = path.join(root, safePath);
      baseDir = root;
    }

    if (!filePath.startsWith(path.resolve(baseDir))) {
      res.statusCode = 403;
      res.end("Forbidden");
      return;
    }

    fs.stat(filePath, (err, stat) => {
      if (err || !stat.isFile()) {
        res.statusCode = 404;
        res.end("Not found");
        return;
      }

      const ext = path.extname(filePath).toLowerCase();
      const contentType = {
        ".ttf": "font/ttf",
        ".woff": "font/woff",
        ".woff2": "font/woff2",
        ".eot": "application/vnd.ms-fontobject",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".css": "text/css",
        ".js": "text/javascript",
        ".html": "text/html",
      }[ext] || "application/octet-stream";

      res.setHeader("Content-Type", contentType);
      const stream = fs.createReadStream(filePath);
      stream.on("error", () => {
        res.statusCode = 500;
        res.end("Error reading file");
      });
      stream.pipe(res);
    });
  });

  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      resolve({ server, port: address.port });
    });
  });
}

function ensureKleRoot(kleRoot) {
  if (!fs.existsSync(kleRoot)) {
    throw new Error(
      `KLE repo not found at '${kleRoot}'.\n` +
        "Clone it with:\n  git clone https://github.com/ijprest/keyboard-layout-editor.git " +
        kleRoot
    );
  }
  const kbHtml = path.join(kleRoot, "kb.html");
  if (!fs.existsSync(kbHtml)) {
    throw new Error(`Missing kb.html in '${kleRoot}'. Is this the keyboard-layout-editor repo?`);
  }
}

function prepareRenderer(kleRoot) {
  const kbHtml = fs.readFileSync(path.join(kleRoot, "kb.html"), "utf8");
  const templates = {
    keycap_html: extractTemplate(kbHtml, "keycap_html"),
    keycap_svg: extractTemplate(kbHtml, "keycap_svg"),
    keyboard_svg: extractTemplate(kbHtml, "keyboard_svg"),
  };

  global.doT = doT;
  global.$ = (selector) => {
    const id = selector.startsWith("#") ? selector.slice(1) : selector;
    return {
      html: () => templates[id] || "",
    };
  };
  global.angular = { toJson: JSON.stringify };

  const context = global;
  const extensions = fs.readFileSync(path.join(kleRoot, "extensions.js"), "utf8");
  vm.runInNewContext(extensions, context, { filename: "extensions.js" });

  const colorSrc = fs.readFileSync(path.join(kleRoot, "js", "color.js"), "utf8");
  vm.runInNewContext(colorSrc, context, { filename: "color.js" });

  const renderKey = require(path.join(kleRoot, "render.js"));
  const serial = require(path.join(kleRoot, "serial.js"));
  renderKey.init();

  return { renderKey, serial };
}

function renderLayout(layout, renderer) {
  const keyboard = renderer.serial.deserialize(layout);
  const sanitize = (value) => value;

  let right = 0;
  let bottom = 0;

  const keyHtml = keyboard.keys
    .map((key) => {
      const html = renderer.renderKey.html(key, sanitize);
      if (key.bbox) {
        right = Math.max(right, key.bbox.x2);
        bottom = Math.max(bottom, key.bbox.y2);
      }
      const profile = key.profile ? ` ${key.profile}` : "";
      return `<div class=\"key${profile}\">${html}</div>`;
    })
    .join("\n");

  const meta = keyboard.meta || {};
  const backcolor = meta.backcolor || "#eeeeee";
  const radii = meta.radii || "6px";
  let backgroundStyle = "";
  if (meta.background) {
    if (typeof meta.background === "string") {
      backgroundStyle = meta.background;
    } else if (typeof meta.background.style === "string") {
      backgroundStyle = meta.background.style;
    }
  }

  return {
    keyHtml,
    width: right,
    height: bottom,
    backcolor,
    radii,
    backgroundStyle,
    customCss: meta.css || "",
  };
}

function buildHtml(payload, css, baseHref, faCssHref, html2canvasHref) {
  return `<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<base href="${baseHref}">
${faCssHref ? `<link rel="stylesheet" href="${faCssHref}">` : ""}
${html2canvasHref ? `<script src="${html2canvasHref}"></script>` : ""}
<style>
${css}
</style>
${payload.customCss ? `<style>${payload.customCss}</style>` : ""}
<style>body{margin:0;background:#fff;}</style>
</head>
<body>
<div id="keyboard">
  <div id="keyboard-bg" style="width:${payload.width}px; height:${payload.height}px; background-color:${payload.backcolor}; border-radius:${payload.radii}; ${payload.backgroundStyle}">
    ${payload.keyHtml}
  </div>
</div>
</body>
</html>`;
}

function writeDataUrlToFile(dataUrl, outputPath) {
  const prefix = "data:image/jpeg;base64,";
  if (!dataUrl.startsWith(prefix)) {
    throw new Error("Unexpected html2canvas output format (expected JPEG data URL).");
  }
  const encoded = dataUrl.slice(prefix.length);
  fs.writeFileSync(outputPath, Buffer.from(encoded, "base64"));
}

function resolveOutputPath(inputPath, args) {
  if (args.output) {
    return args.output;
  }
  const baseName = path.basename(inputPath, path.extname(inputPath));
  if (args.outDir) {
    return path.join(args.outDir, `${baseName}.jpg`);
  }
  return path.join(path.dirname(inputPath), `${baseName}.jpg`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const inputFiles = discoverInputs(args.inputs);

  ensureKleRoot(args.kleRoot);

  if (args.outDir) {
    fs.mkdirSync(args.outDir, { recursive: true });
  }

  const renderer = prepareRenderer(args.kleRoot);
  const [kbCss, webfontCss] = await Promise.all([
    compileStylusFile(path.join(args.kleRoot, "kb.css")),
    compileStylusFile(path.join(args.kleRoot, "kbd-webfont.css")),
  ]);
  const fontAwesome = resolveFontAwesome(args.kleRoot);
  if (!fontAwesome) {
    console.warn("Warning: font-awesome CSS not found. Icons may be missing.");
  }
  const html2canvas = args.renderMode === "kle-jpg" ? resolveHtml2Canvas() : null;
  if (args.renderMode === "kle-jpg" && !html2canvas) {
    throw new Error(
      "html2canvas is required for --render-mode kle-jpg. Install dependencies with: npm install"
    );
  }
  const compiledCss = `${kbCss}\n${webfontCss}`;

  const mounts = [];
  if (fontAwesome) {
    mounts.push({ prefix: "/fa", dir: path.dirname(fontAwesome.mountDir) });
  }
  if (html2canvas) {
    mounts.push({ prefix: "/deps", dir: html2canvas.mountDir });
  }
  const { server, port } = await startStaticServer(args.kleRoot, mounts);
  const baseHref = `http://127.0.0.1:${port}/`;

  const browser = await puppeteer.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({
      width: args.viewport.width,
      height: args.viewport.height,
      deviceScaleFactor: args.scale,
    });

    for (const inputPath of inputFiles) {
      const raw = fs.readFileSync(inputPath, "utf8");
      const layout = JSON.parse(raw);
      const payload = renderLayout(layout, renderer);
      const html = buildHtml(
        payload,
        compiledCss,
        baseHref,
        fontAwesome ? fontAwesome.cssHref : null,
        html2canvas ? html2canvas.scriptHref : null
      );

      await page.setContent(html, { waitUntil: args.waitUntil, timeout: args.navTimeoutMs });
      if (fontAwesome) {
        await page
          .waitForFunction(
            (href) => Array.from(document.styleSheets).some((ss) => ss.href && ss.href.includes(href)),
            { timeout: args.navTimeoutMs || 10000 },
            fontAwesome.cssHref
          )
          .catch(() => undefined);
      }
      if (html2canvas) {
        await page
          .waitForFunction(() => typeof window.html2canvas === "function", {
            timeout: args.navTimeoutMs || 10000,
          })
          .catch(() => undefined);
      }
      if (args.fontWaitMs > 0) {
        await page.evaluate((fontWaitMs, faGlyph) => {
          if (!document.fonts || !document.fonts.ready) return undefined;
          const loads = [];
          // Proactively trigger FontAwesome loads if present.
          loads.push(document.fonts.load("14px FontAwesome", faGlyph));
          loads.push(document.fonts.load('14px "FontAwesome"', faGlyph));
          return Promise.race([
            Promise.all(loads).catch(() => undefined),
            document.fonts.ready,
            new Promise((resolve) => setTimeout(resolve, fontWaitMs)),
          ]);
        }, args.fontWaitMs, FA_TEST_GLYPH);
      }

      const fontCheck = await page.evaluate((faGlyph) => {
        const sample = document.querySelector(".fa");
        const style = sample ? getComputedStyle(sample) : null;
        const family = style ? style.fontFamily : "";
        const faLoaded = document.fonts ? document.fonts.check("14px FontAwesome", faGlyph) : null;
        return { hasSample: !!sample, family, faLoaded };
      }, FA_TEST_GLYPH);
      if (fontCheck.hasSample && !fontCheck.faLoaded) {
        console.warn(
          `Warning: FontAwesome did not load for ${inputPath} (font-family: ${fontCheck.family || "unknown"}).`
        );
      }

      const outPath = resolveOutputPath(inputPath, args);
      if (args.renderMode === "kle-jpg") {
        const jpegDataUrl = await page.evaluate(async (jpegQuality) => {
          function getResizedCanvas(canvas, newWidth, newHeight, bgcolor) {
            const tmpCanvas = document.createElement("canvas");
            tmpCanvas.width = newWidth;
            tmpCanvas.height = newHeight;
            const ctx = tmpCanvas.getContext("2d");
            if (bgcolor !== "") {
              ctx.rect(0, 0, newWidth, newHeight);
              ctx.fillStyle = bgcolor;
              ctx.fill();
            }
            ctx.drawImage(
              canvas,
              0,
              0,
              canvas.width,
              canvas.height,
              0,
              0,
              newWidth,
              newHeight
            );
            return tmpCanvas;
          }

          const keyboardBg = document.querySelector("#keyboard-bg");
          if (!keyboardBg) {
            throw new Error("Unable to find #keyboard-bg after render.");
          }
          if (typeof window.html2canvas !== "function") {
            throw new Error("html2canvas is not available on the page.");
          }

          const canvas = await window.html2canvas(keyboardBg, {
            useCORS: true,
            backgroundColor: null,
            logging: false,
          });
          const jpgCanvas = getResizedCanvas(canvas, canvas.width, canvas.height, "white");
          return jpgCanvas.toDataURL("image/jpeg", jpegQuality);
        }, args.quality / 100.0);
        writeDataUrlToFile(jpegDataUrl, outPath);
      } else {
        const element = await page.$("#keyboard-bg");
        if (!element) {
          throw new Error("Unable to find #keyboard-bg after render.");
        }
        const box = await element.boundingBox();
        if (!box) {
          throw new Error("Unable to measure #keyboard-bg bounding box.");
        }

        await page.screenshot({
          path: outPath,
          type: "jpeg",
          quality: args.quality,
          clip: {
            x: Math.max(0, Math.floor(box.x)),
            y: Math.max(0, Math.floor(box.y)),
            width: Math.ceil(box.width),
            height: Math.ceil(box.height),
          },
        });
      }

      console.log(`Wrote ${outPath}`);
    }
  } finally {
    await browser.close();
    server.close();
  }
}

main().catch((err) => {
  console.error(err.message || err);
  process.exit(1);
});
