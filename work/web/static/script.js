//当文件输入变化时，读取文件并展示到input-image
$(document).ready(function()
{  
    $('#image-upload').change(function()    //用户选择文件后，change()触发
    {   
        var file = this.files[0];  
        var reader = new FileReader();  
        reader.onload = function(e)
        {  
            $('#input-image').attr('src', e.target.result);  
        }  
        reader.readAsDataURL(file);  
    });  
    
    $('#apply-filter').click(function(e)    //用户点击按钮后，click()触发
    {  
        e.preventDefault();  
        var formData = new FormData($('#image-form')[0]);  //FormData表单数据=图片文件+所选风格
        $.ajax({  
            url: '/transfer',  
            type: 'POST',                       //POST方法
            data: formData,  
            contentType: false,  
            processData: false,  
            success: function(response)
            {  
                $('#output-image').attr('src', 'data:image/png;base64,' + response.image_data);  
            },  
            error: function(xhr, status, error)
            {  
                console.error("Error: " + error.message);  
            }  
        });  
    });  
});
