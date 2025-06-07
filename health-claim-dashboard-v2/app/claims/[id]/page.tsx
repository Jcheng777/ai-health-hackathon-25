import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowLeft, FileText } from "lucide-react"

// This would normally come from a database
const claimData = {
  "CL-1234": {
    id: "CL-1234",
    date: "Jun 5, 2025",
    status: "completed",
    riskScore: 28,
    claimText:
      "Patient diagnosed with Type 2 Diabetes (E11.9). Prescribed Metformin 500mg BID. Follow-up visit scheduled in 3 months. Requesting coverage for medication and lab work including HbA1c test.",
    recommendations: ["Include more specific diagnosis codes", "Add documentation for medical necessity"],
  },
  "CL-1233": {
    id: "CL-1233",
    date: "Jun 4, 2025",
    status: "pending",
    riskScore: null,
    claimText:
      "Patient presented with acute sinusitis. Prescribed amoxicillin 500mg TID for 10 days. Requesting coverage for medication and office visit.",
    recommendations: [],
  },
  "CL-1232": {
    id: "CL-1232",
    date: "Jun 3, 2025",
    status: "high-risk",
    riskScore: 76,
    claimText:
      "Patient underwent MRI of lower back due to chronic pain. Diagnosis of lumbar disc herniation. Requesting coverage for MRI and specialist referral.",
    recommendations: [
      "Include prior authorization reference number",
      "Add documentation showing conservative treatment failure",
      "Specify exact CPT code for the MRI procedure",
      "Include radiologist report",
    ],
  },
}

export default function ClaimDetailPage({ params }: { params: { id: string } }) {
  const claim = claimData[params.id as keyof typeof claimData] || {
    id: params.id,
    date: "Unknown",
    status: "unknown",
    riskScore: null,
    claimText: "Claim not found",
    recommendations: [],
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return (
          <span className="rounded-full bg-green-900/50 px-2 py-1 text-xs font-medium text-green-300">Completed</span>
        )
      case "pending":
        return (
          <span className="rounded-full bg-amber-900/50 px-2 py-1 text-xs font-medium text-amber-300">Pending</span>
        )
      case "high-risk":
        return <span className="rounded-full bg-red-900/50 px-2 py-1 text-xs font-medium text-red-300">High Risk</span>
      default:
        return <span className="rounded-full bg-gray-700 px-2 py-1 text-xs font-medium text-gray-300">Unknown</span>
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
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">Claim Details</h1>
            <p className="text-gray-400">
              Claim ID: {claim.id} • Submitted on {claim.date}
            </p>
          </div>
          <div>{getStatusBadge(claim.status)}</div>
        </div>
      </div>

      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Claim Description</CardTitle>
            <CardDescription>Original claim submission</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="rounded-md bg-gray-700 p-4 text-gray-200">
              <FileText className="h-5 w-5 text-gray-400 mb-2" />
              <p>{claim.claimText}</p>
            </div>
          </CardContent>
        </Card>

        {claim.status !== "pending" && (
          <>
            <Card>
              <CardHeader>
                <CardTitle>AI Review Results</CardTitle>
                <CardDescription>Analysis of claim denial risk</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex items-center">
                  <div
                    className={`
                    flex h-24 w-24 items-center justify-center rounded-full 
                    ${
                      claim.riskScore! < 30
                        ? "bg-green-100 text-green-700"
                        : claim.riskScore! < 70
                          ? "bg-amber-100 text-amber-700"
                          : "bg-red-100 text-red-700"
                    }
                  `}
                  >
                    <span className="text-3xl font-bold">{claim.riskScore}%</span>
                  </div>
                  <div className="ml-6">
                    <h3 className="text-lg font-medium text-white">
                      {claim.riskScore! < 30 ? "Low Risk" : claim.riskScore! < 70 ? "Moderate Risk" : "High Risk"}
                    </h3>
                    <p className="text-gray-400">
                      {claim.riskScore! < 30
                        ? "This claim has a good chance of approval."
                        : claim.riskScore! < 70
                          ? "This claim may require additional documentation."
                          : "This claim is likely to be denied without modifications."}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {claim.recommendations.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Recommendations</CardTitle>
                  <CardDescription>Suggested improvements to reduce denial risk</CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {claim.recommendations.map((rec, index) => (
                      <li key={index} className="flex items-start">
                        <span className="mr-2 mt-0.5 text-blue-400">•</span>
                        <span className="text-gray-200">{rec}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            )}
          </>
        )}

        <div className="flex justify-end space-x-4">
          {claim.status === "pending" ? (
            <Button disabled>Awaiting Review</Button>
          ) : (
            <>
              <Button variant="outline">Edit Claim</Button>
              <Button>Submit to Insurance</Button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
