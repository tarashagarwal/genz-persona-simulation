'use client';

import React, { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Heart, Megaphone, AlertTriangle, Info } from "lucide-react";

// shadcn-style components (make sure these exist in src/components/ui/*)
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

// Recharts
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Cell,
} from "recharts";

// ---------- Personas (IDs 0..3 as in your code) ----------
const PERSONAS = [
  {
    id: 0,
    key: "affectionate",
    name: "Affectionate Cheerleader",
    icon: Heart,
    blurb: "Warm, encouraging, supportive tone.",
    img: "/personas/affectionate.jpg",
  },
  {
    id: 1,
    key: "playful",
    name: "Playful Hype-Friend",
    icon: Megaphone,
    blurb: "Bouncy energy, hype and fun.",
    img: "/personas/playful.jpg",
  },
  {
    id: 2,
    key: "anxious",
    name: "Anxious Worrier",
    icon: AlertTriangle,
    blurb: "Cautious, risk-aware, what-if mindset.",
    img: "/personas/anxious.jpg",
  },
  {
    id: 3,
    key: "info",
    name: "Info Recycler",
    icon: Info,
    blurb: "Matter-of-fact, cites what they heard.",
    img: "/personas/info.jpg",
  },
] as const;

// ---------- Fancy chart component (Recharts) ----------
function EmotionChart({
  scores,
  title = "Emotion Scores",
}: {
  scores: Array<{ label: string; confidence: number }>;
  title?: string;
}) {
  if (!Array.isArray(scores) || scores.length === 0) return null;

  // Normalize + sort desc
  const data = scores
    .map((d) => ({
      name: d.label,
      value: Math.max(0, Math.min(1, Number(d.confidence))),
    }))
    .sort((a, b) => b.value - a.value);

  // A small set of pleasant colors; we’ll use gradients for some flair.
  const palette = [
    "#3b82f6", // blue-500
    "#22c55e", // green-500
    "#f59e0b", // amber-500
    "#ef4444", // red-500
    "#a855f7", // violet-500
    "#14b8a6", // teal-500
  ];

  return (
    <Card className="mt-6">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">{title}</CardTitle>
      </CardHeader>
      <CardContent style={{ height: 320 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 16, right: 8, bottom: 8, left: 8 }}>
            {/* SVG gradients (one per bar) */}
            <defs>
              {data.map((d, i) => {
                const color = palette[i % palette.length];
                const id = `grad-${i}`;
                return (
                  <linearGradient key={id} id={id} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={color} stopOpacity={0.95} />
                    <stop offset="100%" stopColor={color} stopOpacity={0.55} />
                  </linearGradient>
                );
              })}
            </defs>

            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} interval={0} height={36} />
            <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} width={44} />
            <Tooltip formatter={(v: number) => `${Math.round(v * 100)}%`} labelFormatter={(l) => `Emotion: ${l}`} />
            <Legend verticalAlign="top" height={24} />
            <Bar dataKey="value" name="Confidence" radius={[10, 10, 0, 0]} isAnimationActive>
              {data.map((_, i) => (
                <Cell key={`cell-${i}`} fill={`url(#grad-${i})`} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="mt-2 text-xs text-slate-500">Values are shown as percent confidence (0–100%).</div>
      </CardContent>
    </Card>
  );
}

export default function PersonaReactionFinder() {
  const [text, setText] = useState("");
  const [personaId, setPersonaId] = useState<number>(PERSONAS[0].id);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<any>(null);

  const selectedPersona = useMemo(
    () => PERSONAS.find((p) => p.id === personaId)!,
    [personaId]
  );

  async function handleFindReaction() {
    setError(null);
    setResponse(null);

    if (!text.trim()) {
      setError("Please paste or type a message first.");
      return;
    }

    setLoading(true);
    try {
      // keep your fetch target as-is
      const res = await fetch("http://localhost:5000/api/find-reaction", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ persona_id: personaId, text }),
      });

      if (!res.ok) {
        const t = await res.text();
        throw new Error(`Request failed (${res.status}): ${t || res.statusText}`);
      }
      const data = await res.json();
      setResponse(data);
    } catch (e: any) {
      setError(e?.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  // Pick the best emotion list to show (prefer emotion_scores, else high_conf_emotions_sent)
  const emotionScores: Array<{ label: string; confidence: number }> =
    (response?.emotion_scores?.length ? response.emotion_scores : response?.high_conf_emotions_sent) || [];

  // Small helpers
  const pct = (v: any) =>
    typeof v === "number" && Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : "—";
  const num = (v: any, d = 3) =>
    typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "—";
  const boolText = (v: any) =>
    typeof v === "boolean" ? (v ? "true" : "false") : String(v ?? "—");

  const attrKeys: string[] = response?.attributes_sent ? Object.keys(response.attributes_sent) : [];

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-white to-slate-50 flex items-center justify-center p-6">
      <div className="w-full max-w-5xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">Persona Reaction Finder</h1>
          <p className="text-slate-600 mt-2">Paste a slogan/notice, pick an archetype, and fetch its reaction.</p>
        </div>

        {/* Message box */}
        <Card className="mb-6">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg">Your message</CardTitle>
          </CardHeader>
          <CardContent>
            <Textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste your slogan or notice here..."
              className="h-28 md:h-32 text-lg"
            />
            <p className="text-xs text-slate-500 mt-2">Tip: Keep it concise so it stays visible in one paste.</p>
          </CardContent>
        </Card>

        {/* Personas */}
        <Card className="mb-6">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg">Pick an archetype</CardTitle>
          </CardHeader>
          <CardContent>
            <RadioGroup
              value={String(personaId)}
              onValueChange={(v) => setPersonaId(Number(v))}
              className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"
            >
              {PERSONAS.map((p) => {
                const Icon = p.icon;
                const active = p.id === personaId;
                return (
                  <label key={p.id} htmlFor={`persona-${p.id}`}>
                    <motion.div
                      whileHover={{ scale: 1.02 }}
                      transition={{ type: "spring", stiffness: 300, damping: 20 }}
                      className={`relative cursor-pointer rounded-2xl border ${
                        active ? "border-slate-900 shadow-lg" : "border-slate-200"
                      } bg-white p-4 h-full flex flex-col`}
                    >
                      <div className="flex items-start gap-3">
                        <div className={`rounded-xl p-2 border ${active ? "border-slate-900" : "border-slate-200"}`}>
                          <Icon className="h-6 w-6" />
                        </div>
                        <div>
                          <div className="font-medium leading-tight">{p.name}</div>
                          <div className="text-xs text-slate-500 mt-1">{p.blurb}</div>
                        </div>
                      </div>

                      {p.img && (
                        <div className="mt-4">
                          <img
                            src={p.img}
                            alt={`${p.name} image`}
                            className="h-24 w-full object-cover rounded-xl border border-slate-200"
                          />
                        </div>
                      )}

                      <div className="mt-4 flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <RadioGroupItem id={`persona-${p.id}`} value={String(p.id)} />
                          <Label htmlFor={`persona-${p.id}`}>Select</Label>
                        </div>
                        <div className={`text-[10px] uppercase tracking-wide ${active ? "text-slate-900" : "text-slate-400"}`}>
                          ID: {p.id}
                        </div>
                      </div>
                    </motion.div>
                  </label>
                );
              })}
            </RadioGroup>
          </CardContent>
        </Card>

        {/* Action */}
        <div className="flex items-center justify-center">
          <Button size="lg" className="rounded-2xl px-8 text-base" onClick={handleFindReaction} disabled={loading}>
            {loading ? "Finding reaction..." : "Find Reaction"}
          </Button>
        </div>

        {/* Error */}
        {error && (
          <div className="mt-6">
            <Card className="border-red-200">
              <CardContent className="pt-6 text-red-600">{error}</CardContent>
            </Card>
          </div>
        )}

        {/* Response */}
        {response && (
          <div className="mt-8 space-y-4">
            {/* Reaction card now shows PARAMETERS/ANALYSIS instead of text */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">Reaction Details</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-600">Persona ID</span>
                    <span className="font-mono">{String(response.persona_id ?? "—")}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Matched Row ID</span>
                    <span className="font-mono">{String(response.matched_row_id ?? "—")}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Similarity</span>
                    <span className="font-mono">{num(response.similarity, 3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Used Attributes</span>
                    <span className="font-mono">{boolText(response.used_attributes)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Masking</span>
                    <span className="font-mono">{boolText(response.masking)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Top Emotion</span>
                    <span className="font-mono">{String(response.top_emotion ?? "—")}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Sentiment</span>
                    <span className="font-mono">{String(response.reddit_sentiment ?? "—")}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Sentiment Score</span>
                    <span className="font-mono">{num(response.score, 2)}</span>
                  </div>
                  <div className="sm:col-span-2">
                    <div className="text-slate-600">Attributes Sent (keys)</div>
                    <div className="mt-1 text-xs font-mono bg-slate-100 border border-slate-200 rounded-lg p-2 max-h-28 overflow-auto">
                      {attrKeys.length > 0 ? attrKeys.join(", ") : "—"}
                    </div>
                  </div>
                  <div className="sm:col-span-2">
                    <div className="text-slate-600">Emotion Scores</div>
                    <div className="mt-2 space-y-1">
                      {emotionScores.length > 0 ? (
                        emotionScores.map((e) => (
                          <div key={e.label} className="flex items-center justify-between text-xs md:text-sm">
                            <span className="capitalize">{e.label}</span>
                            <span className="font-mono">{pct(e.confidence)}</span>
                          </div>
                        ))
                      ) : (
                        <span className="text-xs">—</span>
                      )}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Public vs Private reactions (texts) */}
            {(response?.public_reaction || response?.private_reaction) && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Public (sentiment + masking respected) */}
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">Public Reaction</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-slate-900 text-lg leading-relaxed">
                      {response.public_reaction ?? response.reaction ?? "(no public reaction)"}
                    </div>
                    <div className="mt-3 text-xs text-slate-500 space-x-2">
                      <span>
                        Masking: <span className="font-mono">{boolText(response.masking)}</span>
                      </span>
                      {response.reddit_sentiment && (
                        <span>· Sentiment: <span className="font-mono">{response.reddit_sentiment}</span></span>
                      )}
                      {typeof response.score === "number" && (
                        <span>· Score: <span className="font-mono">{num(response.score, 2)}</span></span>
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* Private (ignores sentiment + masking, emotions only) */}
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">Private Reaction</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-slate-900 text-lg leading-relaxed">
                      {response.private_reaction ?? "(no private reaction)"}
                    </div>
                    <div className="mt-3 text-xs text-slate-500">
                      This is how they’d talk to themselves (unguarded).
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            <Accordion type="single" collapsible>
              <AccordionItem value="json">
                <AccordionTrigger>Show raw JSON</AccordionTrigger>
                <AccordionContent>
                  <Card>
                    <CardContent className="pt-6">
                      <pre className="text-xs md:text-sm overflow-x-auto whitespace-pre-wrap bg-slate-950 text-slate-100 p-4 rounded-xl">
{JSON.stringify(response, null, 2)}
                      </pre>
                    </CardContent>
                  </Card>
                </AccordionContent>
              </AccordionItem>
            </Accordion>

            {/* Fancy Emotion Chart */}
            {(emotionScores?.length ?? 0) > 0 ? (
              <EmotionChart scores={emotionScores} title="Emotion Scores" />
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}
