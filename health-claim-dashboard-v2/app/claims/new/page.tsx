"use client";

import type React from "react";
import { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ArrowLeft, Loader2 } from "lucide-react";
import { toast } from "@/components/ui/use-toast";

// Define insurance types
const insuranceTypes = [
  { value: "self_pay", label: "Self-Pay" },
  { value: "medicare", label: "Medicare" },
  { value: "commercial", label: "Commercial" },
];

export default function NewClaimPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Form state
  const [formData, setFormData] = useState({
    billedAmount: "",
    insuranceType: "",
    providerId: "",
    procedureCode: "",
    diagnosisCode: "",
    dateOfService: "",
    notes: "",
  });

  useEffect(() => {
    const editParam = searchParams.get("edit");
    if (editParam) {
      try {
        const decoded = atob(decodeURIComponent(editParam));
        const parsed = JSON.parse(decoded);
        setFormData({
          billedAmount: parsed.billedAmount || "",
          insuranceType: parsed.insuranceType || "",
          providerId: parsed.providerId || "",
          procedureCode: parsed.procedureCode || "",
          diagnosisCode: parsed.diagnosisCode || "",
          dateOfService: parsed.dateOfService || "",
          notes: parsed.notes || "",
        });
      } catch (e) {
        // ignore if invalid
      }
    }
  }, [searchParams]);

  // Handle input changes
  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  // Handle select changes
  const handleSelectChange = (name: string, value: string) => {
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Basic validation
    if (
      !formData.procedureCode ||
      !formData.diagnosisCode ||
      !formData.insuranceType
    ) {
      toast({
        title: "Missing required fields",
        description: "Please fill in all required fields",
        variant: "destructive",
      });
      return;
    }

    setIsSubmitting(true);

    try {
      // Call /api/predict to get prediction (do not save to DB yet)
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider_id: Number(formData.providerId),
          procedure_code: formData.procedureCode,
          diagnosis_code: formData.diagnosisCode,
          billed_amount: formData.billedAmount
            ? Number(formData.billedAmount)
            : null,
          insurance_type: formData.insuranceType,
          additional_info: formData.notes,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get prediction");
      }

      const result = await response.json();
      // Pass prediction in query param (base64-encoded)
      const reviewData = {
        ...formData,
        prediction: result.prediction,
      };
      const encoded = encodeURIComponent(btoa(JSON.stringify(reviewData)));
      router.push(`/claims/review?data=${encoded}`);
    } catch (error) {
      toast({
        title: "Prediction failed",
        description: "There was an error generating the prediction.",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleSaveClaim = async () => {
    setIsSubmitting(true);
    try {
      // POST claim + prediction data to /api/claims
      const response = await fetch("/api/claims", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider_id: Number(formData.providerId),
          procedure_code: formData.procedureCode,
          diagnosis_code: formData.diagnosisCode,
          billed_amount: formData.billedAmount
            ? Number(formData.billedAmount)
            : null,
          insurance_type: formData.insuranceType,
          additional_info: formData.notes,
        }),
      });
      if (!response.ok) {
        throw new Error("Failed to save claim");
      }
      toast({
        title: "Claim saved!",
        description: "The claim has been saved to the dashboard.",
        variant: "default",
      });
      router.push("/");
    } catch (error) {
      toast({
        title: "Save failed",
        description: "There was an error saving your claim.",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900">
      <div className="container mx-auto max-w-3xl py-8">
        <div className="mb-6">
          <Button variant="ghost" size="sm" asChild className="mb-4">
            <Link href="/">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Dashboard
            </Link>
          </Button>
          <h1 className="text-2xl font-bold text-white">Submit New Claim</h1>
          <p className="text-gray-400">Enter claim details for AI review</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Claim Details Section */}
          <Card>
            <CardHeader>
              <CardTitle>Claim Details</CardTitle>
              <CardDescription>
                Enter the specific information about this insurance claim
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-6">
                {/* First row */}
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="dateOfService">
                      Date of Service <span className="text-red-500">*</span>
                    </Label>
                    <Input
                      id="dateOfService"
                      name="dateOfService"
                      type="date"
                      required
                      value={formData.dateOfService}
                      onChange={handleChange}
                    />
                    <p className="text-xs text-gray-500">Format: YYYY-MM-DD</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="insuranceType">
                      Insurance Type <span className="text-red-500">*</span>
                    </Label>
                    <Select
                      value={formData.insuranceType}
                      onValueChange={(value) =>
                        handleSelectChange("insuranceType", value)
                      }
                      required
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select insurance type" />
                      </SelectTrigger>
                      <SelectContent>
                        {insuranceTypes.map((type) => (
                          <SelectItem key={type.value} value={type.value}>
                            {type.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {/* Second row */}
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="procedureCode">
                      Procedure Code (CPT/HCPCS){" "}
                      <span className="text-red-500">*</span>
                    </Label>
                    <Input
                      id="procedureCode"
                      name="procedureCode"
                      placeholder="e.g., 99213"
                      value={formData.procedureCode}
                      onChange={handleChange}
                      required
                    />
                    <p className="text-xs text-gray-500">
                      Enter the CPT or HCPCS code for the procedure
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="diagnosisCode">
                      Diagnosis Code (ICD-10){" "}
                      <span className="text-red-500">*</span>
                    </Label>
                    <Input
                      id="diagnosisCode"
                      name="diagnosisCode"
                      placeholder="e.g., E11.9"
                      value={formData.diagnosisCode}
                      onChange={handleChange}
                      required
                    />
                    <p className="text-xs text-gray-500">
                      Enter the ICD-10 diagnosis code
                    </p>
                  </div>
                </div>

                {/* Third row */}
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="billedAmount">Billed Amount ($) <span className="text-red-500">*</span></Label>
                    <Input
                      id="billedAmount"
                      name="billedAmount"
                      type="number"
                      step="0.01"
                      min="0"
                      placeholder="0.00"
                      value={formData.billedAmount}
                      onChange={handleChange}
                      required
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="providerId">
                      Provider ID <span className="text-red-500">*</span>
                    </Label>
                    <Input
                      id="providerId"
                      name="providerId"
                      placeholder="Enter provider ID"
                      value={formData.providerId}
                      onChange={handleChange}
                      required
                    />
                    <p className="text-xs text-gray-500">
                      Enter the healthcare provider's ID
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Notes Section */}
          <Card>
            <CardHeader>
              <CardTitle>Additional Notes</CardTitle>
              <CardDescription>
                Add any context, special circumstances, or additional
                information about this claim
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <Textarea
                  id="notes"
                  name="notes"
                  placeholder="Example: Patient has history of similar procedures. Prior authorization was obtained on [date]. Special consideration needed due to emergency circumstances..."
                  className="min-h-[120px]"
                  value={formData.notes}
                  onChange={handleChange}
                />
                <p className="text-xs text-gray-500">
                  Optional: Include any relevant context that might help with
                  the claim review
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Submit Section */}
          <div className="flex justify-between">
            <Button variant="outline" type="button" asChild>
              <Link href="/">Cancel</Link>
            </Button>
            <Button type="submit" disabled={isSubmitting}>
              {isSubmitting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Submitting...
                </>
              ) : (
                "Submit for Review"
              )}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
