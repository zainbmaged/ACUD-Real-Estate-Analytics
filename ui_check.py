import re, json
from playwright.sync_api import sync_playwright
errs=[]; res=[]
def ok(n,c,d=""): res.append(c); print(("PASS " if c else "FAIL ")+n,d)
with sync_playwright() as p:
    b=p.chromium.launch(); 
    ctx=b.new_context(viewport={"width":1300,"height":900}, color_scheme="dark")
    pg=ctx.new_page(); pg.on("pageerror",lambda e:errs.append(str(e))); pg.on("console",lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.goto("http://localhost:8000/"); pg.wait_for_selector("#tabs button"); pg.wait_for_timeout(800)
    ok("starts Arabic RTL", pg.evaluate("[document.documentElement.lang,document.documentElement.dir]")==["ar","rtl"])
    ok("OS dark + auto theme: button offers Light", "Light" in pg.inner_text("#bTheme") or "فاتح" in pg.inner_text("#bTheme"), pg.inner_text("#bTheme"))
    pg.click("#bTheme"); pg.wait_for_timeout(200)
    ok("theme -> light", pg.evaluate("document.documentElement.getAttribute('data-theme')")=="light" and "غامق" in pg.inner_text("#bTheme"), pg.inner_text("#bTheme"))
    bg1=pg.evaluate("getComputedStyle(document.body).backgroundColor")
    pg.click("#bTheme"); pg.wait_for_timeout(200)
    bg2=pg.evaluate("getComputedStyle(document.body).backgroundColor")
    ok("theme -> dark changes colours", pg.evaluate("document.documentElement.getAttribute('data-theme')")=="dark" and bg1!=bg2, f"{bg1} vs {bg2}")
    pg.click("#bLang"); pg.wait_for_timeout(1200)
    ok("lang -> English LTR", pg.evaluate("[document.documentElement.lang,document.documentElement.dir]")==["en","ltr"] and pg.inner_text("#bLang")=="العربية")
    ok("tabs in English", "Executive overview" in pg.inner_text("#tabs"))
    # agents tab
    pg.click('#tabs button[data-k="agents"]'); pg.wait_for_selector("#aGo"); pg.click("#aGo"); pg.wait_for_selector("#aOut .summary",timeout=30000)
    t=pg.inner_text("#aOut"); ok("agent output English (no Arabic)", not re.search("[\u0600-\u06ff]",t), t[:80].replace("\n"," "))
    pg.click("#bLang"); pg.wait_for_selector("#aOut .summary",timeout=30000); pg.wait_for_timeout(1500)
    t=pg.inner_text("#aOut"); ok("switch to Arabic re-runs agent in Arabic", bool(re.search("[\u0600-\u06ff]",t)), t[:60].replace("\n"," "))
    pg.click("#bLang"); pg.wait_for_timeout(1200)
    # every tab, both langs
    for lang in ("en","ar"):
        for k in ["overview","sales","inventory","developers","city","sustain","agents"]:
            pg.click(f'#tabs button[data-k="{k}"]'); pg.wait_for_timeout(900)
            e=pg.query_selector("#main .err"); 
            if e: errs.append(f"{lang}/{k}: "+e.inner_text())
        pg.click("#bLang"); pg.wait_for_timeout(500)
    ok("no page/console errors across 7 tabs x 2 langs", not errs, errs[:3])
    pg.screenshot(path="ui_shot.png")
    b.close()
print(f"{sum(res)}/{len(res)} UI checks passed")
