'use client';

import React, { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Heart, Megaphone, AlertTriangle, Info } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

// --- Personas ---
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

export default function PersonaReactionFinder() {
  const [text, setText] = useState("");
  const [personaId, setPersonaId] = useState<number>(PERSONAS[0].id);  const [loading, setLoading] = useState(false);
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

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-white to-slate-50 flex items-center justify-center p-6">
      <div className="w-full max-w-5xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">Persona Reaction Finder</h1>
          <p className="text-slate-600 mt-2">Paste a slogan/notice, pick an archetype, and fetch its reaction.</p>
        </div>

        
        {/* Text box centered & wide */}
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

        {/* Personas as big selectable tiles */}
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
                      className={
                        `relative cursor-pointer rounded-2xl border ${
                          active ? "border-slate-900 shadow-lg" : "border-slate-200"
                        } bg-white p-4 h-full flex flex-col`
                      }
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
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">Reaction</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-slate-900 text-lg leading-relaxed">{response.reaction || "(no reaction text)"}</div>
                <div className="mt-3 text-xs text-slate-500">
                  Persona ID: <span className="font-mono">{String(response.persona_id ?? "?")}</span> · Similarity: <span className="font-mono">{typeof response.similarity === "number" ? response.similarity.toFixed(3) : String(response.similarity ?? "?")}</span>
                </div>
              </CardContent>
            </Card>

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
          </div>
        )}

              </div>
    </div>
  );
}
