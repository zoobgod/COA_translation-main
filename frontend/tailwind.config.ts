import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bgDeep: "#020203",
        bgBase: "#050506",
        bgElevated: "#0a0a0c",
        fg: "#EDEDEF",
        fgMuted: "#8A8F98",
        accent: "#5E6AD2",
        accentBright: "#6872D9",
      },
      boxShadow: {
        card: "0 0 0 1px rgba(255,255,255,0.06),0 2px 20px rgba(0,0,0,0.4),0 0 40px rgba(0,0,0,0.2)",
        cardHover: "0 0 0 1px rgba(255,255,255,0.1),0 8px 40px rgba(0,0,0,0.5),0 0 80px rgba(94,106,210,0.1)",
        cta: "0 0 0 1px rgba(94,106,210,0.5),0 4px 12px rgba(94,106,210,0.3),inset 0 1px 0 0 rgba(255,255,255,0.2)",
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-20px)" },
        },
        pulseGlow: {
          "0%, 100%": { opacity: "0.11" },
          "50%": { opacity: "0.2" },
        },
        shimmer: {
          "0%": { backgroundPosition: "0% 50%" },
          "100%": { backgroundPosition: "200% 50%" },
        },
      },
      animation: {
        float: "float 10s ease-in-out infinite",
        pulseGlow: "pulseGlow 8s ease-in-out infinite",
        shimmer: "shimmer 5s linear infinite",
      },
    },
  },
  plugins: [],
} satisfies Config;
