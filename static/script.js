// Function to talk to the Gemini Backend

async function askDoctor(defectName, boardContext) {

    const responseDiv = document.getElementById("doctor-response");

    const btn = document.getElementById("askDoctorBtn");



    // 1. Show loading state

    responseDiv.innerHTML = "<i>🧠 Doctor is analyzing " + defectName + "... Please wait.</i>";

    btn.disabled = true;

    btn.style.backgroundColor = "#94a3b8"; 



    try {

        // 2. Call your FastAPI endpoint

        const response = await fetch('/ask-doctor', {

            method: 'POST',

            headers: { 'Content-Type': 'application/json' },

            body: JSON.stringify({

                defect_type: defectName,

                board_type: boardContext,

                component_code: "" 

            })

        });



        const data = await response.json();



        // 3. Display the results

        if (data.status === "success") {

            responseDiv.innerHTML = "<b>📋 Diagnosis for " + data.defect + ":</b><br><br>" + data.advice;

        } else {

            responseDiv.innerHTML = "<span style='color:red;'>Error: " + data.message + "</span>";

        }

    } catch (error) {

        console.error("Error:", error);

        responseDiv.innerHTML = "<span style='color:red;'>⚠️ Failed to contact the doctor. Check terminal for errors.</span>";

    } finally {

        // 4. Reset button

        btn.disabled = false;

        btn.style.backgroundColor = "#3b82f6";

    }

}