// Xử lý đăng nhập
document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault();  // Ngừng hành động mặc định của form

    // Lấy giá trị từ form
    var username = document.getElementById('username').value;
    var password = document.getElementById('password').value;

    console.log('Username:', username);  // Kiểm tra username
    console.log('Password:', password);  // Kiểm tra password

    // Gửi yêu cầu đăng nhập đến Flask backend
    fetch('/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            username: username,
            password: password
        })
    })
    .then(response => {
        console.log('Response status:', response.status);  // Kiểm tra trạng thái HTTP

        if (response.status === 401) {
            alert("Thông tin đăng nhập không đúng. Vui lòng kiểm tra lại.");
            return;  // Dừng việc xử lý tiếp nếu mã lỗi 401
        }

        // Nếu trạng thái là 200, tiến hành xử lý thành công
        return response.json();  // Chuyển phản hồi thành JSON
    })
    .then(data => {
        if (data && data.access_token) {
            console.log('Access Token:', data.access_token);  // Kiểm tra token

            // Lưu trữ token vào localStorage
            localStorage.setItem('access_token', data.access_token);

            // Chuyển hướng đến trang recommend_jobs
            window.location.href = '/recommend_jobs';
        } else {
            alert("Lỗi: Không nhận được access token.");
        }
    })
    .catch(error => {
        console.error('Error:', error);  // Nếu có lỗi trong quá trình fetch
        alert('Đã có lỗi xảy ra. Vui lòng thử lại.');
    });
});

// Kiểm tra token khi truy cập các trang bảo mật
document.addEventListener('DOMContentLoaded', function() {
    const accessToken = localStorage.getItem('access_token');

    if (!accessToken) {
        alert("Token không hợp lệ hoặc chưa đăng nhập!");
        window.location.href = '/login_page';  // Chuyển hướng nếu không có token
    }
});

// Xử lý đăng ký
document.getElementById('registerForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirmPassword').value;

    // Kiểm tra mật khẩu và xác nhận mật khẩu có khớp không
    if (password !== confirmPassword) {
        alert('Mật khẩu và xác nhận mật khẩu không khớp!');
        return;
    }

    // Gửi thông tin đăng ký
    fetch('/register', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ username: username, password: password, confirmPassword: confirmPassword })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Lỗi: ' + data.error);
        } else {
            window.location.href = '/login_page'; // Chuyển hướng đến trang đăng nhập
        }
    })
    .catch(error => {
        console.error('Lỗi:', error);
        alert('Đã có lỗi xảy ra khi đăng ký. Vui lòng thử lại.');
    });
});
