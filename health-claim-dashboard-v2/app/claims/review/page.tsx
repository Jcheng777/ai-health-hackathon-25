"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { ArrowLeft, CheckCircle, Clock, Loader2 } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

type ReviewStatus = "pending" | "analyzing" | "complete"

export default function ClaimReviewPage() {
  const [status, setStatus] = useState<ReviewStatus>("pending")
  const [progress, setProgress] = useState(0)
  const [riskScore, setRiskScore] = useState<number | null>(null)
  const [recommendations, setRecommendations] = useState<string[]>([])

  // Simulate the AI review process
  useEffect(() => {
    const simulateReview = async () => {
      // Start with pending status
      await new Promise((resolve) => setTimeout(resolve, 1500))

      // Move to analyzing status
      setStatus("analyzing")

      // Simulate progress updates
      const interval = setInterval(() => {
        setProgress((prev) => {
          const newProgress = prev + Math.random() * 15
          return newProgress >= 100 ? 100 : newProgress
        })
      }, 600)

      // Complete after some time
      await new Promise((resolve) => setTimeout(resolve, 5000))
      clearInterval(interval)
      setProgress(100)

      // Set final results
      setStatus("complete")
      setRiskScore(42)
      setRecommendations([
        "Include more specific diagnosis codes (ICD-10)",
        "Add documentation for medical necessity",
        "Verify CPT code matches the procedure description",
        "Include prior authorization reference number",
      ])
    }

    simulateReview()
  }, [])

  const renderStatusContent = () => {
    switch (status) {
      case "pending":
        return (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <Clock className="h-16 w-16 text-gray-400 mb-4" />
            <h2 className="text-xl font-semibold text-gray-200">Preparing Review</h2>
            <p className="text-gray-400 mt-2">Your claim is in the queue for AI analysis</p>
            <div className="mt-6">
              <Loader2 className="h-8 w-8 animate-spin text-blue-400 mx-auto" />
            </div>
          </div>
        )

      case "analyzing":
        return (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <Loader2 className="h-16 w-16 text-blue-400 mb-4 animate-spin" />
            <h2 className="text-xl font-semibold text-gray-200">Analyzing Claim</h2>
            <p className="text-gray-400 mt-2">Our AI is reviewing your claim details</p>
            <div className="w-full max-w-md mt-6">
              <Progress value={progress} className="h-2" />
              <p className="text-sm text-gray-400 mt-2">{Math.round(progress)}% complete</p>
            </div>
          </div>
        )

      case "complete":
        return (
          <div className="py-6">
            <div className="flex items-center justify-center mb-8">
              <CheckCircle className="h-12 w-12 text-green-400 mr-4" />
              <div>
                <h2 className="text-xl font-semibold text-gray-200">Analysis Complete</h2>
                <p className="text-gray-400">Review finished on June 7, 2025</p>
              </div>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Deny Risk Score</CardTitle>
                  <CardDescription>Likelihood of claim denial</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-center">
                    <div
                      className={`
                      flex h-36 w-36 items-center justify-center rounded-full 
                      ${
                        riskScore! < 30
                          ? "bg-green-100 text-green-700"
                          : riskScore! < 70
                            ? "bg-amber-100 text-amber-700"
                            : "bg-red-100 text-red-700"
                      }
                    `}
                    >
                      <span className="text-4xl font-bold">{riskScore}%</span>
                    </div>
                  </div>
                  <div className="mt-4 text-center">
                    <p className="text-sm font-medium">
                      {riskScore! < 30
                        ? "Low risk of denial"
                        : riskScore! < 70
                          ? "Moderate risk of denial"
                          : "High risk of denial"}
                    </p>
                  </div>
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
                Review the recommendations and update your claim before final submission to the insurance provider.
              </AlertDescription>
            </Alert>

            <div className="mt-6 flex justify-end space-x-4">
              <Button variant="outline">Edit Claim</Button>
              <Button>Submit to Insurance</Button>
            </div>
          </div>
        )
    }
  }

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
        <p className="text-gray-400">Claim ID: CL-1235 • Submitted on June 7, 2025</p>
      </div>

      <Card>{renderStatusContent()}</Card>
    </div>
  )
}
