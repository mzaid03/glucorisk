"use client";

import { useState } from "react";
import type { FormEvent } from "react";

type FormState = {
  age: string;
  heightFeet: string;
  heightInches: string;
  weightPounds: string;
  waistInches: string;
  systolic: string;
  diastolic: string;
  moderateMinutes: string;
  vigorousMinutes: string;
  sedentaryHours: string;
  sleepHours: string;
  smokingStatus: string;
};

type PredictionResult = {
  risk_score: number;
  risk_percentage: number;
  screening_result:
    | "lower_screening_risk"
    | "higher_screening_risk";
  decision_threshold: number;
  recommendation: string;
  disclaimer: string;
};

const initialForm: FormState = {
  age: "45",
  heightFeet: "5",
  heightInches: "8",
  weightPounds: "194",
  waistInches: "39.4",
  systolic: "132",
  diastolic: "84",
  moderateMinutes: "150",
  vigorousMinutes: "0",
  sedentaryHours: "8",
  sleepHours: "7",
  smokingStatus: "0",
};

export default function Home() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] =
    useState<PredictionResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  function updateField(
    field: keyof FormState,
    value: string,
  ) {
    setForm((current) => ({
      ...current,
      [field]: value,
    }));
  }

  async function handleSubmit(
    event: FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    const totalHeightInches =
      Number(form.heightFeet) * 12 +
      Number(form.heightInches);

    if (totalHeightInches <= 0) {
      setError("Please enter a valid height.");
      setLoading(false);
      return;
    }

    const bmi =
      (703 * Number(form.weightPounds)) /
      totalHeightInches ** 2;

    const payload = {
      age: Number(form.age),
      bmi,
      waist_cm: Number(form.waistInches) * 2.54,
      avg_systolic_bp: Number(form.systolic),
      avg_diastolic_bp: Number(form.diastolic),

      recreation_met_minutes_week:
        Number(form.moderateMinutes) * 4 +
        Number(form.vigorousMinutes) * 8,

      sedentary_minutes:
        Number(form.sedentaryHours) * 60,

      average_sleep_hours: Number(form.sleepHours),
      smoking_status: Number(form.smokingStatus),
    };

    try {
      const apiUrl =
        process.env.NEXT_PUBLIC_API_URL ??
        "http://127.0.0.1:8000";

      const response = await fetch(
        `${apiUrl}/predict`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        },
      );

      if (!response.ok) {
        throw new Error(
          "The server could not process these values.",
        );
      }

      const prediction: PredictionResult =
        await response.json();

      setResult(prediction);
    } catch {
      setError(
        "Unable to reach GlucoRisk. Make sure the FastAPI server is running.",
      );
    } finally {
      setLoading(false);
    }
  }

  const inputStyle =
    "mt-2 w-full rounded-xl border border-slate-300 " +
    "bg-white px-4 py-3 text-slate-900 outline-none " +
    "transition focus:border-teal-600 focus:ring-2 " +
    "focus:ring-teal-100";

  const labelStyle =
    "block text-sm font-semibold text-slate-700";

  return (
    <main className="min-h-screen bg-[#f4f8f7] text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-5">
          <div>
            <span className="text-2xl font-bold text-teal-800">
              GlucoRisk
            </span>
            <p className="text-xs text-slate-500">
              Elevated-A1C screening prototype
            </p>
          </div>

          <span className="rounded-full bg-teal-50 px-4 py-2 text-sm font-medium text-teal-800">
            Research project
          </span>
        </div>
      </header>

      <section className="mx-auto max-w-6xl px-6 py-14">
        <div className="mb-10 max-w-3xl">
          <p className="mb-3 font-semibold uppercase tracking-widest text-teal-700">
            Early screening support
          </p>

          <h1 className="text-4xl font-bold leading-tight md:text-6xl">
            Understand whether an A1C test may be worth
            discussing.
          </h1>

          <p className="mt-6 text-lg leading-8 text-slate-600">
            GlucoRisk uses non-laboratory health information
            to estimate the likelihood of an elevated A1C
            result. It does not diagnose diabetes.
          </p>
        </div>

        <div className="grid gap-8 lg:grid-cols-[1.4fr_0.8fr]">
          <form
            onSubmit={handleSubmit}
            className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm md:p-9"
          >
            <h2 className="text-2xl font-bold">
              Screening questionnaire
            </h2>

            <p className="mt-2 text-slate-500">
              Enter your most recent measurements.
            </p>

            <div className="mt-8 grid gap-6 md:grid-cols-2">
              <label className={labelStyle}>
                Age
                <input
                  className={inputStyle}
                  type="number"
                  min="18"
                  max="80"
                  required
                  value={form.age}
                  onChange={(event) =>
                    updateField("age", event.target.value)
                  }
                />
              </label>

              <label className={labelStyle}>
                Weight in pounds
                <input
                  className={inputStyle}
                  type="number"
                  min="50"
                  max="700"
                  step="0.1"
                  required
                  value={form.weightPounds}
                  onChange={(event) =>
                    updateField(
                      "weightPounds",
                      event.target.value,
                    )
                  }
                />
              </label>

              <label className={labelStyle}>
                Height feet
                <input
                  className={inputStyle}
                  type="number"
                  min="3"
                  max="8"
                  required
                  value={form.heightFeet}
                  onChange={(event) =>
                    updateField(
                      "heightFeet",
                      event.target.value,
                    )
                  }
                />
              </label>

              <label className={labelStyle}>
                Additional height inches
                <input
                  className={inputStyle}
                  type="number"
                  min="0"
                  max="11"
                  required
                  value={form.heightInches}
                  onChange={(event) =>
                    updateField(
                      "heightInches",
                      event.target.value,
                    )
                  }
                />
              </label>

              <label className={labelStyle}>
                Waist circumference in inches
                <input
                  className={inputStyle}
                  type="number"
                  min="16"
                  max="79"
                  step="0.1"
                  required
                  value={form.waistInches}
                  onChange={(event) =>
                    updateField(
                      "waistInches",
                      event.target.value,
                    )
                  }
                />
              </label>

              <label className={labelStyle}>
                Systolic blood pressure
                <input
                  className={inputStyle}
                  type="number"
                  min="60"
                  max="250"
                  required
                  value={form.systolic}
                  onChange={(event) =>
                    updateField(
                      "systolic",
                      event.target.value,
                    )
                  }
                />
              </label>

              <label className={labelStyle}>
                Diastolic blood pressure
                <input
                  className={inputStyle}
                  type="number"
                  min="30"
                  max="150"
                  required
                  value={form.diastolic}
                  onChange={(event) =>
                    updateField(
                      "diastolic",
                      event.target.value,
                    )
                  }
                />
              </label>

              <label className={labelStyle}>
                Moderate exercise minutes per week
                <input
                  className={inputStyle}
                  type="number"
                  min="0"
                  max="3000"
                  required
                  value={form.moderateMinutes}
                  onChange={(event) =>
                    updateField(
                      "moderateMinutes",
                      event.target.value,
                    )
                  }
                />
              </label>

              <label className={labelStyle}>
                Vigorous exercise minutes per week
                <input
                  className={inputStyle}
                  type="number"
                  min="0"
                  max="3000"
                  required
                  value={form.vigorousMinutes}
                  onChange={(event) =>
                    updateField(
                      "vigorousMinutes",
                      event.target.value,
                    )
                  }
                />
              </label>

              <label className={labelStyle}>
                Sedentary hours per day
                <input
                  className={inputStyle}
                  type="number"
                  min="0"
                  max="24"
                  step="0.5"
                  required
                  value={form.sedentaryHours}
                  onChange={(event) =>
                    updateField(
                      "sedentaryHours",
                      event.target.value,
                    )
                  }
                />
              </label>

              <label className={labelStyle}>
                Average sleep hours
                <input
                  className={inputStyle}
                  type="number"
                  min="1"
                  max="24"
                  step="0.5"
                  required
                  value={form.sleepHours}
                  onChange={(event) =>
                    updateField(
                      "sleepHours",
                      event.target.value,
                    )
                  }
                />
              </label>

              <label className={labelStyle}>
                Smoking status
                <select
                  className={inputStyle}
                  value={form.smokingStatus}
                  onChange={(event) =>
                    updateField(
                      "smokingStatus",
                      event.target.value,
                    )
                  }
                >
                  <option value="0">Never smoked</option>
                  <option value="1">Former smoker</option>
                  <option value="2">Current smoker</option>
                </select>
              </label>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="mt-8 w-full rounded-xl bg-teal-700 px-6 py-4 font-bold text-white transition hover:bg-teal-800 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading
                ? "Calculating screening score..."
                : "Calculate screening score"}
            </button>

            {error && (
              <p className="mt-5 rounded-xl bg-red-50 p-4 text-sm text-red-700">
                {error}
              </p>
            )}
          </form>

          <aside className="space-y-6">
            <div className="rounded-3xl bg-slate-900 p-7 text-white">
              <p className="text-sm font-semibold uppercase tracking-widest text-teal-300">
                Your result
              </p>

              {!result ? (
                <p className="mt-5 leading-7 text-slate-300">
                  Complete the questionnaire to generate an
                  estimated elevated-A1C screening score.
                </p>
              ) : (
                <>
                  <p className="mt-5 text-6xl font-bold">
                    {result.risk_percentage}%
                  </p>

                  <p className="mt-3 text-lg font-semibold">
                    {result.screening_result ===
                    "higher_screening_risk"
                      ? "Higher screening risk"
                      : "Lower screening risk"}
                  </p>

                  <p className="mt-5 leading-7 text-slate-300">
                    {result.recommendation}
                  </p>

                  <p className="mt-5 border-t border-slate-700 pt-5 text-xs leading-5 text-slate-400">
                    {result.disclaimer}
                  </p>
                </>
              )}
            </div>

            <div className="rounded-3xl border border-slate-200 bg-white p-7">
              <h2 className="font-bold">Model snapshot</h2>

              <dl className="mt-5 space-y-4 text-sm">
                <div className="flex justify-between">
                  <dt className="text-slate-500">Recall</dt>
                  <dd className="font-semibold">77.2%</dd>
                </div>

                <div className="flex justify-between">
                  <dt className="text-slate-500">ROC-AUC</dt>
                  <dd className="font-semibold">73.5%</dd>
                </div>

                <div className="flex justify-between">
                  <dt className="text-slate-500">Dataset</dt>
                  <dd className="font-semibold">
                    NHANES 2021-2023
                  </dd>
                </div>
              </dl>
            </div>
          </aside>
        </div>
      </section>
    </main>
  );
}