'use client';

import React, { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { Save, Edit, Trash, Paperclip } from "lucide-react";

const ReactQuill = dynamic(() => import("react-quill"), { ssr: false });
import "react-quill/dist/quill.snow.css";

const API_DOMAIN = "https://wp9s6wxn0h.execute-api.us-east-1.amazonaws.com";
const PUBLIC_BUCKET_URL = "https://coltie-uploads.s3.amazonaws.com/uploads/";

export default function NoticeDetailCard({
  initialNotice,
  onToast,
  onDelete,
}: {
  initialNotice: any;
  onToast: (title: string, message: string) => void;
  onDelete: (id: string) => void;
}) {
  const [notice, setNotice] = useState(initialNotice);
  const [editMode, setEditMode] = useState(false);
  const [originalNotice, setOriginalNotice] = useState(initialNotice);
  const [isChanged, setIsChanged] = useState(false);
  const [pendingUploads, setPendingUploads] = useState<File[]>([]);

  useEffect(() => {
    setIsChanged(
      notice.title !== originalNotice.title ||
      notice.description !== originalNotice.description ||
      notice.attachments.length !== originalNotice.attachments.length
    );
  }, [notice, originalNotice]);

  async function handleToggleRead() {
    const updatedIsRead = !notice.isRead;
    try {
      const response = await fetch(`${API_DOMAIN}/trigger-read/${notice.id}`, {
        method: "GET",
      });
      if (!response.ok) {
        throw new Error('Failed to update read status');
      }
      setNotice({ ...notice, isRead: updatedIsRead });
      handleNotification(updatedIsRead ? "Marked as Read" : "Marked as Unread");
    } catch (error) {
      console.error(error);
      handleNotification("Failed to update read status");
    }
  }

  function handleTitleChange(e: React.ChangeEvent<HTMLInputElement>) {
    setNotice({ ...notice, title: e.target.value });
  }

  function handleDescriptionChange(value: string) {
    setNotice({ ...notice, description: value });
  }

  async function handleDelete() {
    if (!window.confirm("Are you sure you want to delete this notice?")) return;
    try {
      const response = await fetch(`${API_DOMAIN}/delete-notice/${notice.id}`, {
        method: "GET",
      });
      if (!response.ok) {
        throw new Error("Failed to delete notice");
      }
      onDelete(notice.id);
      handleNotification("Notice deleted successfully");
    } catch (error) {
      console.error(error);
      handleNotification("Failed to delete notice");
    }
  }

  function handleAttachmentUpload(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files) {
      const file = e.target.files[0];
  
      const newAttachment = {
        id: `attach-${Date.now()}`,
        url: URL.createObjectURL(file),
        type: file.type,
        file: file,
        name: file.name,
      };
  
      setNotice((prevNotice) => ({
        ...prevNotice,
        attachments: [...prevNotice.attachments, newAttachment],
      }));
  
      //setIsChanged(true); // 🚀 Mark it changed when new attachment added!
  
      handleNotification(`File ready to upload: ${file.name}`);
    }
  }
  
  

  function handleAttachmentRemove(id: string) {
    setNotice({ ...notice, attachments: notice.attachments.filter(a => a.id !== id) });
    handleNotification("Attachment removed");
  }

  async function handleSave() {
    try {
      const uploadedAttachments: { fileName: string; type: string }[] = [];
  
      // 1. Upload each new file in attachments
      for (const att of notice.attachments) {
        if (!att.file) continue; // Skip already uploaded ones
  
        // a) Get presigned URL (without noticeId)
        const presignedRes = await fetch(`${API_DOMAIN}/get-signed-url`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            fileType: att.type,
            fileName: att.name,
            noticeId: notice.id
          }),
        });
  
        if (!presignedRes.ok) {
          throw new Error("Failed to get presigned URL");
        }
  
        const presignedData = await presignedRes.json();
  
        // b) Upload the file
        await fetch(presignedData.uploadUrl, {
          method: "PUT",
          headers: { "Content-Type": att.type },
          body: att.file,
        });
        console.log(`File uploaded: ${presignedData}`);
        // c) Save uploaded filename for notice update
        const fileName = att.name
        uploadedAttachments.push({
          fileName: fileName,
          type: att.type,
        });
      }
  
      // 2. Post updated notice
      const response = await fetch(`${API_DOMAIN}/update-notice/${notice.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: notice.title,
          description: notice.description,
          attachments: uploadedAttachments,
        }),
      });
  
      if (!response.ok) {
        throw new Error("Failed to update notice");
      }
  
      // 3. Update frontend
      const updatedFields = await response.json();
      const updatedNotice = {
        ...notice,
        title: updatedFields.title ?? notice.title,
        description: updatedFields.description ?? notice.description,
        attachments: updatedFields.attachments ?? notice.attachments,
      };
  
      setOriginalNotice(updatedNotice);
      setNotice(updatedNotice);
      setEditMode(false);
      setIsChanged(false);
  
      handleNotification("Notice updated successfully");
    } catch (error) {
      console.error(error);
      handleNotification("Failed to update notice");
    }
  }
  

  function handleNotification(message: string) {
    onToast(notice.title, message);
  }

  return (
    <div className="relative p-4 w-full min-w-[600px]">
      <div className="relative border rounded-2xl shadow-md p-6 pt-10 bg-white flex flex-col h-full overflow-hidden">
        {/* Top Color */}
        <div className="absolute top-0 left-0 right-0 h-4 bg-gradient-to-r from-orange-700 to-red-200 rounded-t-2xl"></div>

        {/* Title and Read Toggle */}
        <div className="flex justify-between items-center mb-6">
          <div className="flex-1">
            {editMode ? (
              <input
                type="text"
                value={notice.title}
                onChange={handleTitleChange}
                placeholder="Enter Notice Title..."
                className="border p-2 w-full rounded text-xl font-bold"
              />
            ) : (
              <div className="text-2xl font-bold truncate">{notice.title}</div>
            )}
          </div>
          <div className="flex items-center ml-4">
            <input
              type="checkbox"
              checked={notice.isRead}
              onChange={handleToggleRead}
              className="w-5 h-5 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
            />
          </div>
        </div>

        {/* Description */}
        <div className="mb-4 flex-1">
          <label className="block mb-2 font-bold text-xl">Description</label>
          {editMode ? (
            <ReactQuill
              key="edit"
              value={notice.description}
              onChange={handleDescriptionChange}
              theme="snow"
              modules={{
                toolbar: [
                  [{ size: ['small', false, 'large', 'huge'] }],
                  ['bold', 'italic', 'underline', 'strike'],
                  [{ list: 'ordered' }, { list: 'bullet' }],
                  ['clean'],
                ],
              }}
              formats={[
                'size',
                'bold', 'italic', 'underline', 'strike',
                'list', 'bullet',
              ]}
              className="h-full"
            />
          ) : (
            <div className="ql-editor" dangerouslySetInnerHTML={{ __html: notice.description }} />
          )}
        </div>

        {/* Attachments */}
        <div className="mb-4">
          <div className="flex flex-wrap gap-4 mb-2">
            {notice.attachments.filter((att) => att.type).map(att => (
              <div key={att.id} className="relative">
                
                {att.type.startsWith("image") ? (
                  <img src={att.url} alt="Attachment" className="w-64 h-40 object-cover rounded" />
                ) : att.type.startsWith("video") ? (
                  <video src={att.url} controls preload="metadata" className="w-64 h-40 rounded shadow-md" />
                ) : (
                  <div className="flex flex-col items-center justify-center w-64 h-40 border rounded shadow-md bg-gray-50 text-gray-700 p-4">
                    <span className="text-3xl mb-2">📄</span>
                    <p className="text-xs truncate">{att.url.split('/').pop()}</p>
                    <a href={att.url} download className="text-blue-500 text-xs underline mt-2">
                      Download
                    </a>
                  </div>
                )}
                {editMode && (
                  <button
                    onClick={() => handleAttachmentRemove(att.id)}
                    className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 text-white p-1 rounded-full"
                  >
                    X
                  </button>
                )}
              </div>
            ))}
          </div>

          {/* Upload Button */}
          {editMode && (
            <div className="flex items-center gap-2 mt-4">
              <button
                onClick={() => document.getElementById(`fileInput-${notice.id}`)?.click()}
                className="flex items-center gap-2 bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded mr-2 mt-20"
              >
                <Paperclip className="h-5 w-5" />
              </button>
              <input
                id={`fileInput-${notice.id}`}
                type="file"
                onChange={handleAttachmentUpload}
                className="hidden"
              />
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex justify-end gap-4 mt-6">
          <button
            onClick={() => setEditMode(!editMode)}
            className="flex items-center gap-2 bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded"
          >
            <Edit className="h-5 w-5" />
            {editMode ? "Cancel" : ""}
          </button>
          <button
            onClick={handleSave}
            disabled={!editMode || !isChanged}
            className={`flex items-center gap-2 px-4 py-2 rounded ${
              editMode && isChanged
                ? "bg-indigo-500 hover:bg-indigo-600 text-white"
                : "bg-gray-300 text-gray-500 cursor-not-allowed"
            }`}
          >
            <Save className="h-5 w-5" />
          </button>
        </div>

        {/* Delete */}
        <button
          onClick={handleDelete}
          className="mt-4 flex justify-center items-center bg-red-400 hover:bg-red-600 text-white px-6 py-2 rounded"
        >
          <Trash className="h-5 w-5" />
        </button>
      </div>
    </div>
  );
}
