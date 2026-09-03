"use client";


import { useState } from "react";
import { pipeline } from "@huggingface/transformers";


type SentimentResult = {
  label: string;
  score: number;
};

let classifier: any = null;

export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<SentimentResult | null>(null);
  const [loading, setLoading] = useState(false);

  async function analyzeSentiment() {
    if (!text.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      if (!classifier) {
        classifier = await pipeline(
          "sentiment-analysis",
          "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
        );
      }

      const output = await classifier(text);

      setResult(output[0]);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  }


  return (
    <main className="min-h-screen flex bg-white items-center justify-center p-8">
      <div className="w-full max-w-xl space-y-6">

        <div>
          <h1 className="text-3xl text-black font-semibold">
            Sentiment Analysis 🤗
          </h1>

          <p className="text-gray-600 mt-2">
            Enter a phrase and let the model determine its sentiment.
          </p>
        </div>

        <textarea
          className="w-full border border-gray-200 text-gray-800 rounded-lg p-4"
          rows={5}
          placeholder="Enter a phrase..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />

        <button
          className="w-full cursor-pointer rounded-lg bg-black text-white py-3 disabled:opacity-50"
          onClick={analyzeSentiment}
          disabled={loading || !text.trim()}
        >
          {loading ? "Analyzing..." : "Analyze Sentiment"}
        </button>


        {result && (
          <div className="border border-gray-200 text-gray-900 rounded-lg p-6">
            <p className="text-sm text-gray-500">
              Sentiment
            </p>

            <p className="text-3xl font-bold mt-1">
              {result.label}
            </p>

            <p className="mt-3">
              Confidence:{" "}
              {(result.score * 100).toFixed(1)}%
            </p>
          </div>
        )}

      </div>
    </main>
  );
}