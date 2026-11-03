
import { useState } from "react";
import type { ReactNode } from "react";

type LandingPageProps = {
  onSelectResident: () => void;
  onSelectInvestor: () => void;
  onSelectGovernment: () => void;
};

const BACKGROUND_IMAGE = "/acud-capital.jpg";

const ACUD_LOGO = "/acud-logo.png";
const URBANSIGHT_LOGO = "/urbansight-logo.jpg";

const A2O_URL = "#";

type DashboardKey =
  | "overview"
  | "forecast"
  | "sales"
  | "service"
  | "sustain"
  | "agents";

type Dashboard = {
  key: DashboardKey;
  href: string;
  icon: ReactNode;
};

const dashboards: Dashboard[] = [
  {
    key: "overview",
    href: "/NAC_Real_Estate_Intelligence_Standalone.html",
    icon: (
      <>
        <path d="M3 3h8v8H3z" />
        <path d="M13 3h8v5h-8z" />
        <path d="M13 10h8v11h-8z" />
        <path d="M3 13h8v8H3z" />
      </>
    ),
  },
  {
    key: "forecast",
    href: "http://localhost:8080/public/dashboard2.html#roi",
    icon: (
      <>
        <path d="M3 3v18h18" />
        <path d="M7 15l4-4 3 3 5-6" />
      </>
    ),
  },
  {
    key: "sales",
    href: "http://127.0.0.1:8081/residential-sales-map.html",
    icon: (
      <>
        <path d="M12 21s7-6.2 7-11a7 7 0 1 0-14 0c0 4.8 7 11 7 11Z" />
        <circle cx="12" cy="10" r="2.5" />
      </>
    ),
  },
  {
    key: "service",
    href: "http://127.0.0.1:8081/acud-executive-dashboard.html",
    icon: (
      <>
        <path d="M4 5h16v11H8l-4 4V5Z" />
        <path d="M8 9h8M8 12h5" />
      </>
    ),
  },
  {
    key: "sustain",
    href: "http://127.0.0.1:8000/agents-ui",
    icon: (
      <>
        <path d="M5 19c0-8 5-13 14-14 0 9-5 14-13 14M5 19c1-4 4-7 8-9" />
      </>
    ),
  },
  {
    key: "agents",
    href: A2O_URL,
    icon: (
      <>
        <circle cx="12" cy="12" r="3" />
        <path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2" />
      </>
    ),
  },
];

export default function LandingPage({
  onSelectResident,
  onSelectInvestor,
  onSelectGovernment,
}: LandingPageProps) {
  const [lang, setLang] = useState<"en" | "ar">("en");

  const ar = lang === "ar";

  const copy = {
    tagline: ar
      ? "العاصمة الجديدة، مستقبل أفضل"
      : "NEW ADMINISTRATIVE CAPITAL",

    eyebrow: ar
      ? "عاصمتنا، مسؤوليتنا المشتركة"
      : "YOUR CAPITAL. YOUR VOICE.",

    title: ar
      ? "معًا نبني عاصمة أفضل"
      : "Together, we build a better capital.",

    description: ar
      ? "بوابتك الرقمية للتواصل مع شركة العاصمة الإدارية للتنمية العمرانية، وتقديم الشكاوى والاستفسارات."
      : "Your digital gateway to ACUD for submitting complaints, inquiries, and service requests.",

    choose: ar ? "كيف يمكننا مساعدتك؟" : "How can we help you?",

    residents: ar ? "السكان" : "Residents",

    residentDesc: ar
      ? "قدّم شكوى بشأن الطرق أو الإنارة أو النظافة أو المياه أو مواقف السيارات أو الخدمات داخل الحي."
      : "Report issues with roads, lighting, waste, water, parking, and neighborhood services.",

    residentButton: ar ? "تقديم شكوى" : "Submit a complaint",

    investors: ar ? "المستثمرون" : "Investors",

    investorDesc: ar
      ? "تواصل بشأن الفرص الاستثمارية والمشروعات والاستفسارات والخدمات المتعلقة بالمستثمرين."
      : "Contact us about investment opportunities, projects, and investor-related inquiries.",

    investorButton: ar ? "خدمات المستثمرين" : "Investor services",

    government: ar ? "الخدمات الحكومية" : "Government Services",

    governmentDesc: ar
      ? "استكشف الخدمات الحكومية والإجراءات الرقمية، واحصل على المعلومات اللازمة للاستفسارات والخدمات المتاحة."
      : "Explore government services and digital procedures, and find information about available services and inquiries.",

    governmentButton: ar
      ? "استعراض الخدمات الحكومية"
      : "Explore government services",

    footer: ar
      ? "شركة العاصمة الإدارية للتنمية العمرانية · مصر"
      : "Administrative Capital for Urban Development · Egypt",

    servicesLabel: ar ? "الخدمات الرقمية" : "ACUD DIGITAL SERVICES",

    servicesHint: ar
      ? "اختر الفئة المناسبة للمتابعة إلى الخدمة المطلوبة."
      : "Choose the option that best describes your needs.",

    future: ar ? "المستقبل يبدأ هنا" : "THE FUTURE STARTS HERE",

    dashLabel: ar ? "لوحات التحليلات" : "ANALYTICS & INTELLIGENCE",

    dashTitle: ar ? "لوحات المعلومات الذكية" : "Intelligence dashboards",

    dashHint: ar
      ? "نظرة عامة، تنبؤات، خرائط مبيعات، ذكاء الخدمات، والاستدامة، وتوصيات الوكلاء."
      : "Explore the complete real estate overview, forecasting, sales mapping, service intelligence, sustainability and AI agent recommendations.",

    open: ar ? "فتح اللوحة" : "Open dashboard",

    dash: {
      overview: {
        tag: ar ? "نظرة عامة" : "Overview",
        name: ar
          ? "منظومة ذكاء العاصمة الإدارية"
          : "NAC Real Estate Intelligence Overview",
        desc: ar
          ? "استكشف المنظومة المتكاملة لذكاء العاصمة الإدارية، مع رؤى القطاع العقاري والمؤشرات والتحليلات الرئيسية."
          : "Explore the complete New Administrative Capital intelligence platform, bringing together real estate insights, key indicators and strategic analytics.",
      },

      forecast: {
        tag: ar ? "التنبؤ" : "Forecasting",
        name: ar
          ? "لوحة التعلم الآلي والتنبؤ"
          : "NAC ML & Forecasting Dashboard",
        desc: ar
          ? "نماذج التعلم الآلي والتنبؤات وعائد الاستثمار للعاصمة الإدارية."
          : "Machine-learning models, forecasts and ROI for the New Administrative Capital.",
      },

      sales: {
        tag: ar ? "خريطة المبيعات" : "Sales map",
        name: ar
          ? "خريطة مبيعات الوحدات السكنية"
          : "ACUD | Residential Sales Map",
        desc: ar
          ? "استكشف مبيعات الوحدات السكنية جغرافيًا عبر الأحياء."
          : "Explore residential sales geographically across districts.",
      },

      service: {
        tag: ar ? "ذكاء الخدمات" : "Service intelligence",
        name: ar
          ? "ذكاء خدمات السكان"
          : "ACUD Resident Service Intelligence",
        desc: ar
          ? "لوحة تنفيذية لشكاوى السكان وأداء الخدمات."
          : "Executive view of resident complaints and service performance.",
      },

      sustain: {
        tag: ar ? "الاستدامة" : "Sustainability",
        name: ar
          ? "ACUD · ذكاء القطاع العقاري"
          : "ACUD · Real Estate Intelligence",
        desc: ar
          ? "رؤى الاستدامة والقطاع العقاري مدعومة بالوكلاء الأذكياء."
          : "Sustainability and real-estate insights powered by AI agents.",
      },

      agents: {
        tag: ar ? "توصيات الوكلاء" : "Agents recommendation",
        name: "A2O ELITE — Decision Studio",
        desc: ar
          ? "استوديو القرار: توصيات الوكلاء الأذكياء لدعم اتخاذ القرار."
          : "Decision studio with agent-generated recommendations.",
      },
    },
  };

  return (
    <main
      dir={ar ? "rtl" : "ltr"}
      className="min-h-screen bg-[#EEEEEE] text-[#1D1616]"
    >
      {/* HEADER */}
      <header className="bg-[#1D1616] text-white">
        <nav className="mx-auto flex max-w-7xl items-center justify-between px-5 py-5 md:px-10">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-white px-3 py-1.5">
              <img
                src={ACUD_LOGO}
                alt="ACUD — Administrative Capital for Urban Development"
                className="h-12 w-auto"
              />
            </div>

            <div className="rounded-lg bg-white px-2 py-1">
              <img
                src={URBANSIGHT_LOGO}
                alt="UrbanSight — Smarter insights. Stronger decisions."
                className="h-12 w-auto"
              />
            </div>
          </div>

          <button
            onClick={() => setLang(ar ? "en" : "ar")}
            className="rounded-lg border border-white/30 px-4 py-2 text-sm transition hover:bg-white/10"
            aria-label={ar ? "Switch to English" : "التبديل إلى العربية"}
          >
            {ar ? "English" : "العربية"}
          </button>
        </nav>
      </header>

      {/* HERO */}
      <section
        className="relative overflow-hidden bg-[#1D1616] bg-cover bg-center text-white"
        style={{ backgroundImage: `url(${BACKGROUND_IMAGE})` }}
      >
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-r from-[#1D1616]/90 via-[#1D1616]/70 to-[#8E1616]/40" />

        <div className="pointer-events-none absolute -right-24 -top-24 h-80 w-80 rounded-full bg-[#8E1616]/40 blur-3xl" />

        <div className="relative mx-auto grid max-w-7xl items-center gap-10 px-5 py-16 md:grid-cols-2 md:px-10 md:py-28">
          <div>
            <p className="mb-5 text-sm font-semibold uppercase tracking-[0.2em] text-[#D84040]">
              {copy.eyebrow}
            </p>

            <h1 className="max-w-2xl text-4xl font-extrabold leading-tight md:text-6xl">
              {copy.title}
            </h1>

            <p className="mt-6 max-w-xl text-base leading-8 text-gray-200 md:text-lg">
              {copy.description}
            </p>

            <button
              onClick={() =>
                document.getElementById("services")?.scrollIntoView({
                  behavior: "smooth",
                })
              }
              className="mt-8 rounded-lg bg-[#D84040] px-6 py-3 font-semibold text-white transition hover:bg-[#b92e2e]"
            >
              {copy.choose} ↓
            </button>
          </div>

          <div className="relative">
            <div className="flex aspect-[5/4] items-center justify-center overflow-hidden rounded-2xl border border-white/20 bg-black/30 p-8 backdrop-blur-sm">
              <div className="flex flex-col items-center gap-4 text-center">
                <div className="rounded-xl bg-white px-4 py-2">
                  <img
                    src={URBANSIGHT_LOGO}
                    alt="UrbanSight"
                    className="h-28 w-auto md:h-44"
                  />
                </div>

                <div className="mx-auto mt-1 h-1 w-24 rounded-full bg-[#D84040]" />

                <p className="text-sm tracking-[0.25em] text-white/80">
                  {copy.future}
                </p>
              </div>
            </div>

            <div className="absolute -bottom-4 -left-4 h-20 w-20 rounded-xl border border-[#D84040]/50" />
          </div>
        </div>
      </section>

      {/* DASHBOARDS */}
      <section
        id="dashboards"
        className="mx-auto max-w-7xl px-5 py-16 md:px-10 md:py-20"
      >
        <div className="mx-auto mb-10 max-w-2xl text-center">
          <p className="text-sm font-bold uppercase tracking-widest text-[#8E1616]">
            {copy.dashLabel}
          </p>

          <h2 className="mt-3 text-3xl font-extrabold md:text-4xl">
            {copy.dashTitle}
          </h2>

          <p className="mt-3 text-gray-600">{copy.dashHint}</p>
        </div>

        <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
          {dashboards.map(({ key, href, icon }) => {
            const d = copy.dash[key];

            return (
              <a
                key={key}
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className={`group flex flex-col rounded-2xl border bg-white p-7 shadow-sm transition hover:-translate-y-1 hover:shadow-xl ${
                  key === "overview"
                    ? "border-[#8E1616]/40 shadow-md"
                    : "border-gray-200 hover:border-[#8E1616]/40"
                }`}
              >
                <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-[#8E1616]/10 text-[#8E1616]">
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.7"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="h-7 w-7"
                    aria-hidden="true"
                  >
                    {icon}
                  </svg>
                </div>

                <span className="mt-5 text-xs font-bold uppercase tracking-widest text-[#D84040]">
                  {d.tag}
                </span>

                <h3 className="mt-2 text-xl font-bold">{d.name}</h3>

                <p className="mt-3 flex-1 leading-7 text-gray-600">
                  {d.desc}
                </p>

                <span className="mt-6 inline-flex items-center gap-2 font-semibold text-[#8E1616] transition group-hover:gap-3">
                  {copy.open}
                  <span aria-hidden="true">{ar ? "←" : "→"}</span>
                </span>
              </a>
            );
          })}
        </div>
      </section>

      {/* DIGITAL SERVICES */}
      <section
        id="services"
        className="mx-auto max-w-7xl px-5 pb-16 md:px-10 md:pb-24"
      >
        <div className="mx-auto mb-10 max-w-2xl text-center">
          <p className="text-sm font-bold uppercase tracking-widest text-[#8E1616]">
            {copy.servicesLabel}
          </p>

          <h2 className="mt-3 text-3xl font-extrabold md:text-4xl">
            {copy.choose}
          </h2>

          <p className="mt-3 text-gray-600">{copy.servicesHint}</p>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          {/* RESIDENTS */}
          <article className="group flex flex-col rounded-2xl border border-gray-200 bg-white p-7 shadow-sm transition hover:-translate-y-1 hover:border-[#8E1616]/40 hover:shadow-xl md:p-8">
            <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-[#8E1616]/10 text-[#8E1616]">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                className="h-7 w-7"
                aria-hidden="true"
              >
                <path d="M16 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                <circle cx="10" cy="7" r="4" />
                <path d="M20 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" />
              </svg>
            </div>

            <h3 className="mt-6 text-2xl font-bold">
              {copy.residents}
            </h3>

            <p className="mt-4 flex-1 leading-7 text-gray-600">
              {copy.residentDesc}
            </p>

            <button
              onClick={onSelectResident}
              className="mt-8 flex w-full items-center justify-center gap-2 rounded-lg bg-[#8E1616] px-5 py-4 font-semibold text-white transition hover:bg-[#6f1010]"
            >
              {copy.residentButton}
              <span aria-hidden="true">{ar ? "←" : "→"}</span>
            </button>
          </article>

          {/* INVESTORS */}
          <article className="group flex flex-col rounded-2xl border border-gray-200 bg-white p-7 shadow-sm transition hover:-translate-y-1 hover:border-[#D84040]/50 hover:shadow-xl md:p-8">
            <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-[#D84040]/10 text-[#8E1616]">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                className="h-7 w-7"
                aria-hidden="true"
              >
                <path d="M3 21h18M5 21V7l8-4v18M19 21V11l-6-4" />
                <path d="M8 9v.01M8 12v.01M8 15v.01M8 18v.01M16 13v.01M16 16v.01M16 19v.01" />
              </svg>
            </div>

            <h3 className="mt-6 text-2xl font-bold">
              {copy.investors}
            </h3>

            <p className="mt-4 flex-1 leading-7 text-gray-600">
              {copy.investorDesc}
            </p>

            <button
              onClick={onSelectInvestor}
              className="mt-8 flex w-full items-center justify-center gap-2 rounded-lg bg-[#1D1616] px-5 py-4 font-semibold text-white transition hover:bg-[#3c2929]"
            >
              {copy.investorButton}
              <span aria-hidden="true">{ar ? "←" : "→"}</span>
            </button>
          </article>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="bg-[#1D1616] px-5 py-7 text-center text-sm text-gray-300">
        <p>{copy.footer}</p>

        <p className="mt-2 text-xs text-gray-500">
          © {new Date().getFullYear()} ACUD
        </p>
      </footer>
    </main>
  );
}
