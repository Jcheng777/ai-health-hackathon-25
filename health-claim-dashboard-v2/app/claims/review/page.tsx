"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ArrowLeft, CheckCircle, Clock, Loader2 } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

type ReviewStatus = "pending" | "analyzing" | "complete";

export default function ClaimReviewPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const [status, setStatus] = useState<ReviewStatus>("pending");
  const [progress, setProgress] = useState(0);
  const [predictionObj, setPredictionObj] = useState<any>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [likelihoodPercent, setLikelihoodPercent] = useState<number | null>(
    null
  );
  const [denialReasons, setDenialReasons] = useState<string[] | null>(null);
  const [nextSteps, setNextSteps] = useState<string[] | null>(null);
  const [analysisDetails, setAnalysisDetails] = useState<any>(null);
  const [acceptedReasons, setAcceptedReasons] = useState<string[] | null>(null);
  const [reviewData, setReviewData] = useState<any>(null);

  useEffect(() => {
    const dataParam = searchParams.get("data");
    if (dataParam) {
      try {
        const decoded = atob(decodeURIComponent(dataParam));
        const parsed = JSON.parse(decoded);
        setReviewData(parsed);
        // Extract prediction fields for display
        const predictionJson = parsed.prediction || {};
        setPrediction(predictionJson.prediction ?? null);
        setConfidence(
          predictionJson.confidence_percent !== undefined
            ? predictionJson.confidence_percent
            : null
        );
        setLikelihoodPercent(
          predictionJson.likelihood_percent !== undefined
            ? predictionJson.likelihood_percent
            : null
        );
        setDenialReasons(predictionJson.denial_reasons ?? null);
        setNextSteps(predictionJson.next_steps ?? null);
        setAnalysisDetails(predictionJson.analysis_details ?? null);
        setAcceptedReasons(predictionJson.accepted_reasons ?? null);
        setStatus("complete");
        setProgress(100);
      } catch (e) {
        setStatus("pending");
      }
    }
  }, [searchParams]);

  if (!prediction) {
    return (
      <div className="container mx-auto max-w-3xl py-8">
        <div className="mb-6">
          <Button variant="ghost" size="sm" asChild className="mb-4">
            <Link href="/">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Dashboard
            </Link>
          </Button>
          <h1 className="text-2xl font-bold text-white">Claim Review</h1>
          <p className="text-gray-400">
            No prediction data found. Please submit a claim first.
          </p>
        </div>
      </div>
    );
  }

  const handleSaveClaim = async () => {
    if (!reviewData) return;
    setIsSaving(true);
    try {
      let dateCreated = null;
      const predictionJson = reviewData.prediction || {};
      // Try to get date_of_service from form or prediction
      const dateOfService = reviewData.dateOfService || reviewData.date_of_service || (predictionJson.analysis_details && predictionJson.analysis_details.date_of_service);
      if (dateOfService) {
        dateCreated = String(dateOfService).slice(0, 10); // always a string
      } else {
        const today = new Date();
        dateCreated = today.toISOString().slice(0, 10); // always a string
      }
      // Merge form data and prediction fields
      const payload = {
        provider_id: reviewData.providerId || reviewData.provider_id,
        procedure_code: reviewData.procedureCode || reviewData.procedure_code,
        diagnosis_code: reviewData.diagnosisCode || reviewData.diagnosis_code,
        billed_amount: reviewData.billedAmount || reviewData.billed_amount,
        insurance_type: reviewData.insuranceType || reviewData.insurance_type,
        additional_info: reviewData.notes || reviewData.additional_info,
        ...predictionJson,
        date_created: dateCreated,
      };
      const response = await fetch("/api/claims", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error("Failed to save claim");
      }
      router.push("/");
    } catch (error) {
      setIsSaving(false);
    }
  };

  const handleEditClaim = () => {
    if (!reviewData) return;
    // Remove the prediction field before sending back
    const { prediction, ...formFields } = reviewData;
    const encoded = encodeURIComponent(btoa(JSON.stringify(formFields)));
    router.push(`/claims/new?edit=${encoded}`);
  };

  const renderStatusContent = () => {
    switch (status) {
      case "pending":
        return (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <Clock className="h-16 w-16 text-gray-400 mb-4" />
            <h2 className="text-xl font-semibold text-gray-200">
              Preparing Review
            </h2>
            <p className="text-gray-400 mt-2">
              Your claim is in the queue for AI analysis
            </p>
            <div className="mt-6">
              <Loader2 className="h-8 w-8 animate-spin text-blue-400 mx-auto" />
            </div>
          </div>
        );

      case "analyzing":
        return (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <Loader2 className="h-16 w-16 text-blue-400 mb-4 animate-spin" />
            <h2 className="text-xl font-semibold text-gray-200">
              Analyzing Claim
            </h2>
            <p className="text-gray-400 mt-2">
              Our AI is reviewing your claim details
            </p>
            <div className="w-full max-w-md mt-6">
              <Progress value={progress} className="h-2" />
              <p className="text-sm text-gray-400 mt-2">
                {Math.round(progress)}% complete
              </p>
            </div>
          </div>
        );

      case "complete":
        // Format prediction label
        const formattedPrediction = prediction
          ? prediction
              .replace(/_/g, " ")
              .replace(/\b\w/g, (c) => c.toUpperCase())
          : "";
        return (
          <div className="py-6">
            <div className="flex items-center justify-center mb-8">
              <CheckCircle className="h-12 w-12 text-green-400 mr-4" />
              <div>
                <h2 className="text-xl font-semibold text-gray-200">
                  Analysis Complete
                </h2>
              </div>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>AI Denial Prediction</CardTitle>
                </CardHeader>
                <CardContent>
                  {typeof prediction === "string" &&
                  (confidence !== null || likelihoodPercent !== null) ? (
                    <div className="flex flex-col items-center justify-center">
                      <div className="mb-2 text-2xl font-bold">
                        Prediction: {formattedPrediction}
                      </div>
                      <div className="flex h-36 w-36 items-center justify-center rounded-full bg-blue-100 text-blue-700">
                        <span className="text-4xl font-bold">
                          {prediction === "accepted" && confidence !== null
                            ? `${confidence}%`
                            : prediction === "denied" &&
                              likelihoodPercent !== null
                            ? `${likelihoodPercent}%`
                            : ""}
                        </span>
                      </div>
                      <div className="mt-4 text-center">
                        <p className="text-sm font-medium">
                          {prediction === "accepted" && confidence !== null
                            ? "Likelihood of acceptance"
                            : prediction === "denied" &&
                              likelihoodPercent !== null
                            ? "Likelihood of denial"
                            : ""}
                        </p>
                        {prediction === "accepted" &&
                          acceptedReasons &&
                          acceptedReasons.length > 0 && (
                            <div className="mt-2">
                              <strong>Accepted Reasons:</strong>
                              <ul className="space-y-2">
                                {acceptedReasons.map((reason, idx) => (
                                  <li key={idx} className="flex items-start">
                                    <span className="mr-2 mt-0.5 text-blue-400">
                                      •
                                    </span>
                                    <span className="text-gray-200">
                                      {reason}
                                    </span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        {prediction === "denied" &&
                          denialReasons &&
                          denialReasons.length > 0 && (
                          <div className="mt-2">
                            <strong>Denial Reasons:</strong>
                              <ul className="space-y-2">
                              {denialReasons.map((reason, idx) => (
                                  <li key={idx} className="flex items-start">
                                    <span className="mr-2 mt-0.5 text-blue-400">
                                      •
                                    </span>
                                    <span className="text-gray-200">
                                      {reason}
                                    </span>
                                  </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="text-center text-gray-500">
                      No prediction available.
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Recommendations</CardTitle>
                  <CardDescription>Suggested improvements</CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {nextSteps && nextSteps.length > 0 ? (
                      nextSteps.map((step, index) => (
                        <li key={index} className="flex items-start">
                          <span className="mr-2 mt-0.5 text-blue-400">•</span>
                          <span className="text-gray-200">{step}</span>
                        </li>
                      ))
                    ) : (
                      <li className="text-gray-400">
                        No next steps available.
                      </li>
                    )}
                  </ul>
                </CardContent>
              </Card>
            </div>

            <Alert className="mt-6 bg-blue-950/50 border-blue-800">
              <AlertTitle className="text-blue-300">Next Steps</AlertTitle>
              <AlertDescription className="text-gray-300">
                Review the recommendations and update your claim before final
                submission to the insurance provider.
              </AlertDescription>
            </Alert>

            <div className="mt-6 flex justify-end space-x-4">
              <Button variant="outline" onClick={handleEditClaim}>
                Edit Claim
              </Button>
              <Button onClick={handleSaveClaim} disabled={isSaving}>
                {isSaving ? (
                  <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Saving...</>
                ) : (
                  "Save Claim"
                )}
              </Button>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="container mx-auto max-w-3xl py-8">
      <div className="mb-6">
        <Button variant="ghost" size="sm" asChild className="mb-4">
          <Link href="/">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Dashboard
          </Link>
        </Button>
        <h1 className="text-2xl font-bold text-white">Claim Review</h1>

      </div>

      <Card>{renderStatusContent()}</Card>
    </div>
  );
}
