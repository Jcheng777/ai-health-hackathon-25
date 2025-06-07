"use client"

import type React from "react"

import { useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { ArrowLeft, Loader2, Upload, X, FileText } from "lucide-react"
import { toast } from "@/components/ui/use-toast"

export default function NewClaimPage() {
  const router = useRouter()
  const [notes, setNotes] = useState("")
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isDragOver, setIsDragOver] = useState(false)

  const handleFileUpload = (files: FileList | null) => {
    if (!files) return

    const newFiles = Array.from(files).filter((file) => {
      // Accept common document formats
      const allowedTypes = [
        "application/pdf",
        "image/jpeg",
        "image/png",
        "image/tiff",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      ]
      return allowedTypes.includes(file.type)
    })

    if (newFiles.length !== files.length) {
      toast({
        title: "Some files were skipped",
        description: "Only PDF, images, and Word documents are supported",
        variant: "destructive",
      })
    }

    setUploadedFiles((prev) => [...prev, ...newFiles])
  }

  const removeFile = (index: number) => {
    setUploadedFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    handleFileUpload(e.dataTransfer.files)
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes"
    const k = 1024
    const sizes = ["Bytes", "KB", "MB", "GB"]
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (uploadedFiles.length === 0) {
      toast({
        title: "No files uploaded",
        description: "Please upload at least one claim document",
        variant: "destructive",
      })
      return
    }

    setIsSubmitting(true)

    // Simulate API call
    setTimeout(() => {
      setIsSubmitting(false)
      toast({
        title: "Claim submitted",
        description: `Your claim with ${uploadedFiles.length} document(s) has been submitted for AI review`,
      })
      router.push("/claims/review")
    }, 1500)
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
        <h1 className="text-2xl font-bold text-white">Submit New Claim</h1>
        <p className="text-gray-400">Upload claim documents and add any additional notes</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* File Upload Section */}
        <Card>
          <CardHeader>
            <CardTitle>Claim Documents</CardTitle>
            <CardDescription>
              Upload claim forms, medical records, or supporting documentation (PDF, images, Word docs)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div
              className={`
                border-2 border-dashed rounded-lg p-8 text-center transition-colors
                ${isDragOver ? "border-blue-400 bg-blue-950/50" : "border-gray-600 hover:border-gray-500"}
              `}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
              <div className="space-y-2">
                <p className="text-lg font-medium text-gray-200">Drop files here or click to upload</p>
                <p className="text-sm text-gray-400">Supports PDF, JPG, PNG, TIFF, DOC, DOCX up to 10MB each</p>
              </div>
              <input
                type="file"
                multiple
                accept=".pdf,.jpg,.jpeg,.png,.tiff,.doc,.docx"
                onChange={(e) => handleFileUpload(e.target.files)}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
            </div>

            {/* Uploaded Files List */}
            {uploadedFiles.length > 0 && (
              <div className="mt-6 space-y-2">
                <h4 className="font-medium text-gray-200">Uploaded Files ({uploadedFiles.length})</h4>
                {uploadedFiles.map((file, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-700 rounded-md">
                    <div className="flex items-center space-x-3">
                      <FileText className="h-5 w-5 text-gray-400" />
                      <div>
                        <p className="text-sm font-medium text-gray-200">{file.name}</p>
                        <p className="text-xs text-gray-400">{formatFileSize(file.size)}</p>
                      </div>
                    </div>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => removeFile(index)}
                      className="text-gray-400 hover:text-red-400"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Notes Section */}
        <Card>
          <CardHeader>
            <CardTitle>Additional Notes</CardTitle>
            <CardDescription>
              Add any context, special circumstances, or additional information about this claim
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <Textarea
                id="notes"
                placeholder="Example: Patient has history of similar procedures. Prior authorization was obtained on [date]. Special consideration needed due to emergency circumstances..."
                className="min-h-[120px]"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
              />
              <p className="text-xs text-gray-500">
                Optional: Include any relevant context that might help with the claim review
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Submit Section */}
        <div className="flex justify-between">
          <Button variant="outline" type="button" asChild>
            <Link href="/">Cancel</Link>
          </Button>
          <Button type="submit" disabled={isSubmitting || uploadedFiles.length === 0}>
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
  )
}
