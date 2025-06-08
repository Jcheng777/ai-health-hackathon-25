"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
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
import { useRouter } from "next/navigation";

type ReviewStatus = "pending" | "analyzing" | "complete";

export default function ClaimReviewPage() {
  const [status, setStatus] = useState<ReviewStatus>("pending");
  const [progress, setProgress] = useState(0);
  const [riskScore, setRiskScore] = useState<number | null>(null);
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);

  // Simulate the AI review process
  useEffect(() => {
    const simulateReview = async () => {
      await new Promise((resolve) => setTimeout(resolve, 1500));
      setStatus("analyzing");
      const interval = setInterval(() => {
        setProgress((prev) => {
          const newProgress = prev + Math.random() * 15;
          return newProgress >= 100 ? 100 : newProgress;
        });
      }, 600);
      await new Promise((resolve) => setTimeout(resolve, 2000));
      clearInterval(interval);
      setProgress(100);
      setStatus("complete");
      // Read prediction/confidence from router state
      // @ts-ignore
      if (window.history.state && window.history.state.usr) {
        // @ts-ignore
        setPrediction(window.history.state.usr.prediction ?? null);
        // @ts-ignore
        setConfidence(window.history.state.usr.confidence ?? null);
      }
    };
    simulateReview();
  }, []);

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
        return (
          <div className="py-6">
            <div className="flex items-center justify-center mb-8">
              <CheckCircle className="h-12 w-12 text-green-400 mr-4" />
              <div>
                <h2 className="text-xl font-semibold text-gray-200">
                  Analysis Complete
                </h2>
                <p className="text-gray-400">Review finished on June 7, 2025</p>
              </div>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>AI Denial Prediction</CardTitle>
                  <CardDescription>Likelihood of claim denial</CardDescription>
                </CardHeader>
                <CardContent>
                  {prediction !== null && confidence !== null ? (
                    <div className="flex flex-col items-center justify-center">
                      <div className="mb-2 text-2xl font-bold">
                        Prediction: {prediction}
                      </div>
                      <div className="flex h-36 w-36 items-center justify-center rounded-full bg-blue-100 text-blue-700">
                        <span className="text-4xl font-bold">
                          {(confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="mt-4 text-center">
                        <p className="text-sm font-medium">
                          Confidence in prediction
                        </p>
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
                    {recommendations.map((rec, index) => (
                      <li key={index} className="flex items-start">
                        <span className="mr-2 mt-0.5 text-blue-400">•</span>
                        <span className="text-gray-200">{rec}</span>
                      </li>
                    ))}
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
              <Button variant="outline">Edit Claim</Button>
              <Button>Submit to Insurance</Button>
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
        <p className="text-gray-400">
          Claim ID: CL-1235 • Submitted on June 7, 2025
        </p>
      </div>

      <Card>{renderStatusContent()}</Card>
    </div>
  );
}
