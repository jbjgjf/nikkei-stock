import fs from "node:fs";
import path from "node:path";

const BASE = "https://eegs.env.go.jp";

// 控えめに（相手サイト負荷対策）
const WAIT_MS = 120;
const CONCURRENCY = 1; // まずは1推奨。速くしたければ2〜3まで。

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// HTMLから<option value="...">ラベル</option> を全部抜く（軽量パーサ）
function extractOptions(html) {
    const options = [];
    const selectBlocks = html.match(/<select[\s\S]*?<\/select>/gi) || [];
    for (const block of selectBlocks) {
        const re = /<option[^>]*value="([^"]+)"[^>]*>([\s\S]*?)<\/option>/gi;
        let m;
        while ((m = re.exec(block))) {
            const gasID = m[1]?.trim();
            const label = m[2]?.replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();
            if (gasID && gasID !== "") options.push({ gasID, metric: label });
        }
    }
    // 重複除去（gasID基準）
    const seen = new Set();
    return options.filter(o => (seen.has(o.gasID) ? false : (seen.add(o.gasID), true)));
}

function extractGraphJson(html) {
    const m = html.match(/<script[^>]*id="graph"[^>]*>([\s\S]*?)<\/script>/i);
    if (!m) return null;
    const jsonText = m[1].trim();
    return JSON.parse(jsonText);
}

async function fetchText(url) {
    const res = await fetch(url, {
        headers: {
            "User-Agent": "Mozilla/5.0 (compatible; data-collector/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status} ${url}`);
    return await res.text();
}

async function scrapeOne(corporateUrl, outDir) {
    const u = new URL(corporateUrl);
    const spEmitCode = u.searchParams.get("spEmitCode");
    const repDivID = u.searchParams.get("repDivID") || "1";
    if (!spEmitCode) throw new Error(`spEmitCode missing: ${corporateUrl}`);

    // 1) ページHTMLからプルダウン全取得
    const pageHtml = await fetchText(corporateUrl);
    const options = extractOptions(pageHtml);
    if (options.length === 0) throw new Error(`No options found: ${corporateUrl}`);

    // 2) 各gasIDでgraph取得 → #graph JSON
    const rows = [];
    for (const opt of options) {
        const graphUrl =
            `${BASE}/ghg-santeikohyo-result/corporate/graph?` +
            `gasID=${encodeURIComponent(opt.gasID)}&spEmitCode=${encodeURIComponent(spEmitCode)}&repDivID=${encodeURIComponent(repDivID)}`;

        const html = await fetchText(graphUrl);

        let data;
        try {
            data = extractGraphJson(html);
        } catch (e) {
            console.warn("JSON parse failed:", spEmitCode, repDivID, opt, e.message);
            continue;
        }
        if (!data) continue;

        const years = data.repYear || [];
        const vals = data.emitAmount || [];
        const unit = data.unit || "";

        for (let i = 0; i < years.length; i++) {
            rows.push({
                spEmitCode,
                repDivID,
                metric: opt.metric,
                gasID: opt.gasID,
                year: years[i],
                value: vals[i],
                unit,
            });
        }

        await sleep(WAIT_MS);
    }

    // 3) CSV出力
    const header = ["spEmitCode", "repDivID", "metric", "gasID", "year", "value", "unit"];
    const csv = [
        header.join(","),
        ...rows.map(r => header.map(k => JSON.stringify(r[k] ?? "")).join(",")),
    ].join("\n");

    const outPath = path.join(outDir, `eegs_${spEmitCode}_repDiv${repDivID}_yearly.csv`);
    fs.writeFileSync(outPath, csv, "utf8");
    return { spEmitCode, repDivID, options: options.length, rows: rows.length, outPath };
}

async function main() {
    const urlsPath = process.argv[2] || "urls.txt";
    const outDir = process.argv[3] || "out";
    fs.mkdirSync(outDir, { recursive: true });

    const urls = fs.readFileSync(urlsPath, "utf8")
        .split("\n")
        .map(s => s.trim())
        .filter(Boolean)
        .filter(s => !s.startsWith("#"));

    if (urls.length === 0) {
        console.error("No URLs found in", urlsPath);
        process.exit(1);
    }

    console.log(`Targets: ${urls.length}`);
    const results = [];

    // 超シンプルに直列（まずはこれが安全）
    for (const url of urls) {
        try {
            console.log("Scraping:", url);
            const r = await scrapeOne(url, outDir);
            results.push(r);
            console.log("OK:", r.spEmitCode, "rows=", r.rows, "options=", r.options, "->", r.outPath);
        } catch (e) {
            console.error("FAIL:", url, e.message);
        }
    }

    // まとめログ
    const summaryPath = path.join(outDir, "summary.json");
    fs.writeFileSync(summaryPath, JSON.stringify(results, null, 2), "utf8");
    console.log("Done. Summary:", summaryPath);
}

main().catch(e => {
    console.error(e);
    process.exit(1);
});